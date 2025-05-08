import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



# 超图卷积对比学习
class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgnn1 = HGNN_conv(in_dim, hidden_list[0])

    def forward(self, x, G):
        x_embed = self.hgnn1(x, G)
        x_embed_1 = F.leaky_relu(x_embed, 0.25)
        return x_embed_1


class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha=0.5):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)
        self.hgcn2 = HGCN(in_size, hid_list)
        self.fc1 = torch.nn.Linear(hid_list[-1], num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, hid_list[-1])
        self.tau = 0.5
        self.alpha = alpha

    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)
        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)
        loss = self.alpha * self.sim(h1, h2) + (1 - self.alpha) * self.sim(h2, h1)
        return z1, z2, loss


    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    # 求余弦相似度
    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.norm_sim(z1, z1))
        between_sim = f(self.norm_sim(z1, z2))
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.mean()
        return loss


class IntraClassContrastive_m(nn.Module):
    def __init__(self, m_num, dis_num, hidd_list, num_proj_hidden):
        super(IntraClassContrastive_m, self).__init__()
        self.CL_HGCN_m = CL_HGCN(m_num + dis_num, hidd_list, num_proj_hidden)

    def forward(self, concat_mi_tensor, G_m_Kn, G_m_Km):
        mi_embedded = concat_mi_tensor
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_m(mi_embedded, G_m_Kn, mi_embedded, G_m_Km)
        return mi_feature1, mi_feature2, mi_cl_loss


class IntraClassContrastive_d(nn.Module):
    def __init__(self, m_num, dis_num, hidd_list, num_proj_hidden):
        super(IntraClassContrastive_d, self).__init__()
        self.CL_HGCN_dis = CL_HGCN(dis_num + m_num, hidd_list, num_proj_hidden)

    def forward(self, concat_dis_tensor, G_dis_Kn, G_dis_Km):
        dis_embedded = concat_dis_tensor
        dis_feature1, dis_feature2, dis_cl_loss = self.CL_HGCN_dis(dis_embedded, G_dis_Kn, dis_embedded, G_dis_Km)

        return dis_feature1, dis_feature2, dis_cl_loss


# 外部对比学习

def norm_sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


def sim(z1, z2):
    f = lambda x: torch.exp(x / 0.5)
    between_sim = f(norm_sim(z1, z2))
    loss = -torch.log(between_sim.diag() / between_sim.sum(1))
    loss = loss.mean()
    return loss





# 正交融合
class FeatureFusion(nn.Module):
    def __init__(self, col_num):
        super(FeatureFusion, self).__init__()
        self.fc1 = nn.Linear(4 * col_num, 2 * col_num)
        self.fc2 = nn.Linear(4 * col_num, 2 * col_num)

    def forward(self, fea1, fea2, fea3):
        fea_mid = torch.cat((fea1, fea2), dim=1)
        ortloss1 = self.ort_loss(fea1, fea2)
        fea_mid_original_shape = self.fc1(fea_mid)

        fea = torch.cat((fea_mid_original_shape, fea3), dim=1)
        ortloss2 = self.ort_loss(fea_mid_original_shape, fea3)
        fea_original_shape = self.fc2(fea)
        ortloss = ortloss1 + ortloss2
        return fea_original_shape, ortloss

    def ort_loss(self, fea1, fea2):
        all_f = norm_sim(fea1,fea2)
        aa = all_f.diagonal()
        l1_loss = torch.mean(torch.abs(aa))
        l2_loss = torch.mean(torch.pow(aa, 2))
        return l1_loss+l2_loss

# 渐进正交融合
class CyclicFusion(nn.Module):
    def __init__(self, cyclic_num, col_num):
        super(CyclicFusion, self).__init__()
        self.cyclic_num = cyclic_num
        self.fusionlayer = FeatureFusion(col_num)
        self.col_num = col_num

    def forward(self, fea1, fea2, fea3):
        context = torch.zeros([fea1.shape[0], 2 * self.col_num])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        context = context.to(device)
        loss = 0

        for _ in range(self.cyclic_num):
            feain1 = fea1 + context
            feain2 = fea2 + context
            feain3 = fea3 + context
            context, loss_ort = self.fusionlayer(feain1, feain2, feain3)
            loss = loss + loss_ort

        return context, loss



# 双线性解码器
class Decoder(nn.Module):

    def __init__(self, dropout_prob,input):
        super(Decoder, self).__init__()

        self.w = torch.nn.Parameter(torch.randn(input, input, dtype=torch.float32), requires_grad=True)
        self.w = nn.init.xavier_normal_(self.w)

        self.dropout = nn.Dropout(dropout_prob)  # 添加 Dropout 层
        self.sig = nn.Sigmoid()

    def forward(self,jz_fm,jz_fd):
        supp1 = torch.mm(jz_fm, self.w)
        supp1 = self.dropout(supp1)
        decoder = torch.mm(supp1, jz_fd.transpose(0, 1))
        decoder = self.sig(decoder)

        return decoder


class POMSL(nn.Module):
    def __init__(self, m_num, dis_num, hidd_list, num_proj_hidden, mlp_drop,cycle_number):
        super(POMSL, self).__init__()
        # 超图卷积对比学习
        self.InterCL_MSS = IntraClassContrastive_m(m_num, dis_num, hidd_list, num_proj_hidden)
        self.InterCL_MGIP = IntraClassContrastive_m(m_num, dis_num, hidd_list, num_proj_hidden)
        self.InterCL_MSIE = IntraClassContrastive_m(m_num, dis_num, hidd_list, num_proj_hidden)

        self.InterCL_DSS = IntraClassContrastive_d(m_num, dis_num, hidd_list, num_proj_hidden)
        self.InterCL_DGIP = IntraClassContrastive_d(m_num, dis_num, hidd_list, num_proj_hidden)
        self.InterCL_DSIE = IntraClassContrastive_d(m_num, dis_num, hidd_list, num_proj_hidden)

        # 多模态融合
        self.dFeatureFusion = CyclicFusion(cycle_number, hidd_list[0])
        self.mFeatureFusion = CyclicFusion(cycle_number, hidd_list[0])
        # 残差
        self.fc1 = nn.Linear(6 * hidd_list[0], 2 * hidd_list[0])
        self.fc2 = nn.Linear(6 * hidd_list[0], 2 * hidd_list[0])
        # 分类
        self.bilineardecoder = Decoder(mlp_drop,4*hidd_list[0])


    def forward(self, MSS_X, MGIP_X, MSIE_X, DSS_X, DGIP_X, DSIE_X, G_m_MSS_Kn, G_m_MSS_Km, G_m_MGIP_Kn, G_m_MGIP_Km,
                G_m_MSIE_Kn, G_m_MSIE_Km, G_dis_DSS_Kn, G_dis_DSS_Km, G_dis_DGIP_Kn, G_dis_DGIP_Km, G_dis_DSIE_Kn,
                G_dis_DSIE_Km, train_sam):
        MSS_X_IN_KNN, MSS_X_IN_KMEANS, m_cl_MSS_loss = self.InterCL_MSS(MSS_X, G_m_MSS_Kn, G_m_MSS_Km)
        MGIP_X_IN_KNN, MGIP_X_IN_KMEANS, m_cl_MGIP_loss = self.InterCL_MGIP(MGIP_X, G_m_MGIP_Kn, G_m_MGIP_Km)
        MSIE_X_IN_KNN, MSIE_X_IN_KMEANS, m_cl_MSIE_loss = self.InterCL_MSIE(MSIE_X, G_m_MSIE_Kn, G_m_MSIE_Km)

        MSS_X_IN = torch.cat((MSS_X_IN_KNN, MSS_X_IN_KMEANS), dim=1)
        MGIP_X_IN = torch.cat((MGIP_X_IN_KNN, MGIP_X_IN_KMEANS), dim=1)
        MSIE_X_IN = torch.cat((MSIE_X_IN_KNN, MSIE_X_IN_KMEANS), dim=1)
        m_loss_cl_IN = m_cl_MSS_loss + m_cl_MGIP_loss + m_cl_MSIE_loss


        m_loss_MGIPMSS = sim(MSS_X_IN, MGIP_X_IN)
        m_loss_MGIPMSIE = sim(MSIE_X_IN, MGIP_X_IN)
        m_loss_cl_OUT = m_loss_MGIPMSS + m_loss_MGIPMSIE

        DSS_X_IN_KNN, DSS_X_IN_KMEANS, dis_cl_DSS_loss = self.InterCL_DSS(DSS_X, G_dis_DSS_Kn, G_dis_DSS_Km)
        DGIP_X_IN_KNN, DGIP_X_IN_KMEANS, dis_cl_DGIP_loss = self.InterCL_DGIP(DGIP_X, G_dis_DGIP_Kn, G_dis_DGIP_Km)
        DSIE_X_IN_KNN, DSIE_X_IN_KMEANS, dis_cl_DSIE_loss = self.InterCL_DSIE(DSIE_X, G_dis_DSIE_Kn, G_dis_DSIE_Km)

        DSS_X_IN = torch.cat((DSS_X_IN_KNN, DSS_X_IN_KMEANS), dim=1)
        DGIP_X_IN = torch.cat((DGIP_X_IN_KNN, DGIP_X_IN_KMEANS), dim=1)
        DSIE_X_IN = torch.cat((DSIE_X_IN_KNN, DSIE_X_IN_KMEANS), dim=1)
        d_loss_cl_IN = dis_cl_DSS_loss + dis_cl_DGIP_loss + dis_cl_DSIE_loss


        d_loss_DGIPDSS = sim(DSS_X_IN, DGIP_X_IN)
        d_loss_DGIPDSIE = sim(DSIE_X_IN, DGIP_X_IN)
        d_loss_cl_OUT = d_loss_DGIPDSS + d_loss_DGIPDSIE



        # 正交融合
        MM, loss_m_ort = self.mFeatureFusion(MSS_X_IN, MSIE_X_IN, MGIP_X_IN)
        DD, loss_d_ort = self.dFeatureFusion(DSS_X_IN, DSIE_X_IN, DGIP_X_IN)
        MM2 = torch.cat((MSS_X_IN, MSIE_X_IN), dim=1)
        MM21 = torch.cat((MM2, MGIP_X_IN), dim=1)
        MM22 = self.fc1(MM21)
        DD2 = torch.cat((DSS_X_IN, DSIE_X_IN), dim=1)
        DD21 = torch.cat((DD2, DGIP_X_IN), dim=1)
        DD22 = self.fc2(DD21)
        m_feature = torch.cat((MM, MM22), dim=1)
        d_feature = torch.cat((DD, DD22), dim=1)



        score = self.bilineardecoder(m_feature,d_feature)

        return m_loss_cl_IN + d_loss_cl_IN + m_loss_cl_OUT + d_loss_cl_OUT + loss_m_ort + loss_d_ort, score

