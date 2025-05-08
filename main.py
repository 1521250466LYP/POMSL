from sklearn.model_selection import KFold
from getsample import *
import warnings
from torch import *
import torch.optim as optim
from utils import *
from getsimilarity import *
from ConstructHW import *
from model import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fold = 0
n_splits = 5

k_neigs_m = [2]
clusters_m = [500]
k_neigs_d = [2]
clusters_d =[120]

m_num = 2262
d_num = 216
hidden_list = [256]
num_proj_hidden = 64

epoch = 700
mlp_drop = 0.3
lr=0.00005
cycle_number = 2


association = pd.read_csv("M_D.csv", index_col=0).to_numpy()
samples = get_all_samples(association)  # 构建样本集
sum_valid_result = [0, 0, 0, 0, 0, 0]
MGIP, MSIE, MSS, DSS, DGIP, DSIE = get_sim()

kf = KFold(n_splits=n_splits, shuffle=True)  # 进行5折交叉验证
for train_index, val_index in kf.split(samples):
    fold += 1
    train_samples = samples[train_index, :]
    val_samples = samples[val_index, :]

    # 盖住测试集
    for i in val_samples:
        association[i[0], i[1]] = 0



    model = POMSL(m_num, d_num, hidden_list, num_proj_hidden, mlp_drop,cycle_number)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0005)
    model.train()
    # 拼接关联矩阵
    MGIP_X = np.hstack([association, MGIP])
    MSIE_X = np.hstack([association, MSIE])
    MSS_X = np.hstack([association, MSS])
    DSS_X = np.hstack([association.T, DSS])
    DGIP_X = np.hstack([association.T, DGIP])
    DSIE_X = np.hstack([association.T, DSIE])

    MGIP_X = torch.tensor(MGIP_X, dtype=torch.float32)
    MSIE_X = torch.tensor(MSIE_X, dtype=torch.float32)
    MSS_X = torch.tensor(MSS_X, dtype=torch.float32)
    DSS_X = torch.tensor(DSS_X, dtype=torch.float32)
    DGIP_X = torch.tensor(DGIP_X, dtype=torch.float32)
    DSIE_X = torch.tensor(DSIE_X, dtype=torch.float32)

    MGIP_X = MGIP_X.to(device)
    MSIE_X = MSIE_X.to(device)
    MSS_X = MSS_X.to(device)
    DSS_X = DSS_X.to(device)
    DGIP_X = DGIP_X.to(device)
    DSIE_X = DSIE_X.to(device)

    # 构造超图
    G_m_Kn_MGIP = constructHW_knn(MGIP_X.detach().cpu().numpy(), K_neigs=k_neigs_m, is_probH=False)
    G_m_Km_MGIP = constructHW_kmean(MGIP_X.detach().cpu().numpy(), clusters=clusters_m)
    G_m_Kn_MGIP = G_m_Kn_MGIP.to(device)
    G_m_Km_MGIP = G_m_Km_MGIP.to(device)

    G_m_Kn_MSIE = constructHW_knn(MSIE_X.detach().cpu().numpy(), K_neigs=k_neigs_m, is_probH=False)
    G_m_Km_MSIE = constructHW_kmean(MSIE_X.detach().cpu().numpy(), clusters=clusters_m)
    G_m_Kn_MSIE = G_m_Kn_MSIE.to(device)
    G_m_Km_MSIE = G_m_Km_MSIE.to(device)

    G_m_Kn_MSS = constructHW_knn(MSS_X.detach().cpu().numpy(), K_neigs=k_neigs_m, is_probH=False)
    G_m_Km_MSS = constructHW_kmean(MSS_X.detach().cpu().numpy(), clusters=clusters_m)
    G_m_Kn_MSS = G_m_Kn_MSS.to(device)
    G_m_Km_MSS = G_m_Km_MSS.to(device)

    G_d_Kn_DSS = constructHW_knn(DSS_X.detach().cpu().numpy(), K_neigs=k_neigs_d, is_probH=False)
    G_d_Km_DSS = constructHW_kmean(DSS_X.detach().cpu().numpy(), clusters=clusters_d)
    G_d_Kn_DSS = G_d_Kn_DSS.to(device)
    G_d_Km_DSS = G_d_Km_DSS.to(device)

    G_d_Kn_DGIP = constructHW_knn(DGIP_X.detach().cpu().numpy(), K_neigs=k_neigs_d, is_probH=False)
    G_d_Km_DGIP = constructHW_kmean(DGIP_X.detach().cpu().numpy(), clusters=clusters_d)
    G_d_Kn_DGIP = G_d_Kn_DGIP.to(device)
    G_d_Km_DGIP = G_d_Km_DGIP.to(device)

    G_d_Kn_DSIE = constructHW_knn(DSIE_X.detach().cpu().numpy(), K_neigs=k_neigs_d, is_probH=False)
    G_d_Km_DSIE = constructHW_kmean(DSIE_X.detach().cpu().numpy(), clusters=clusters_d)
    G_d_Kn_DSIE = G_d_Kn_DSIE.to(device)
    G_d_Km_DSIE = G_d_Km_DSIE.to(device)

    # 取出训练测试标签和位置
    train_sample = (train_samples[:, 0:2]).astype(np.int64)
    train_lable = (train_samples[:, 2:]).astype(np.float32)
    valid_sample = (val_samples[:, 0:2]).astype(np.int64)
    valid_lable = (val_samples[:, 2:]).astype(np.float32)

    for i in range(0, epoch):
        # model.train()
        loss1 = nn.BCELoss()
        tra_label_new = (torch.from_numpy(train_lable)).cuda().to(torch.float64)
        loss_model, score = model(MSS_X, MGIP_X, MSIE_X, DSS_X, DGIP_X, DSIE_X, G_m_Kn_MSS, G_m_Km_MSS,
                                  G_m_Kn_MGIP, G_m_Km_MGIP, G_m_Kn_MSIE, G_m_Km_MSIE, G_d_Kn_DSS, G_d_Km_DSS,
                                  G_d_Kn_DGIP, G_d_Km_DGIP, G_d_Kn_DSIE, G_d_Km_DSIE, train_sample)
        optimizer.zero_grad()

        aa = score[tuple(train_sample.T)]
        los = loss1(score[tuple(train_sample.T)], (tra_label_new.squeeze(dim=-1)).to(torch.float32))
        # print("交叉熵损失", los)
        loss = loss_model + los
        loss.backward()
        optimizer.step()
        train_score = aa.cpu().detach().numpy()
        print(i + 1)
        result = caculate_metric(train_score, train_lable)

        print(loss)

    model.eval()

    with torch.no_grad():
        _, val_score = model(MSS_X, MGIP_X, MSIE_X, DSS_X, DGIP_X, DSIE_X, G_m_Kn_MSS, G_m_Km_MSS,
                             G_m_Kn_MGIP, G_m_Km_MGIP, G_m_Kn_MSIE, G_m_Km_MSIE, G_d_Kn_DSS, G_d_Km_DSS,
                             G_d_Kn_DGIP, G_d_Km_DGIP, G_d_Kn_DSIE, G_d_Km_DSIE, valid_sample)
        pre = val_score[tuple(valid_sample.T)]
        pre = pre.cpu().detach().numpy()
        print("验证集")
    valid_result = caculate_metric(pre, valid_lable)

    sum_valid_result = [x + y for x, y in zip(sum_valid_result, valid_result)]
print("平均结果")
sum_valid_result = [x / 5 for x in sum_valid_result]
print_met(sum_valid_result)
