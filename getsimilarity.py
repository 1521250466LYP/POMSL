import numpy as np
import pandas as pd


def get_sim ():

    # 读取三个相似性矩阵

    MGIP = pd.read_csv("metabolite_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    MSIE = pd.read_csv("metabolites_information_entropy_similarity.csv", index_col=0,
                         dtype=np.float32).to_numpy()
    MSS = pd.read_csv("metabolites_structure_similarity.csv", index_col=0, dtype=np.float32).to_numpy()



    DSS = pd.read_csv("disease_semantic_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    DGIP = pd.read_csv("disease_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    DSIE = pd.read_csv("disease _information_entropy_similarity .csv", index_col=0,
                         dtype=np.float32).to_numpy()





    return MGIP, MSIE ,MSS ,DSS ,DGIP ,DSIE



