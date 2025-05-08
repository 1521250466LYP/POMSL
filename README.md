# POMSL：Progressive Orthogonal Multimodal Similarity Learning for Metabolite-Disease Association Prediction

# Running process: Simply execute the main.py script.

# requirement

numpy                     1.23.5
pandas                    2.2.2
python                    3.9.19
scipy                     1.13.1
torch                     2.2.1+cu121              
scikit-learn              1.5.1

# File Introduction:
1. `ConstructHW.py` Construct hypergraphs.
2. `getsample.py` Obtain the positive and negative sample sets.
3. `getsimilarity.py` Obtain multiple similarities.
4. `hypergraph_construct_KNN.py` Construct a hypergraph using KNN.
5. `hypergraph_construct_kmeans.py` Construct a hypergraph using K-means.
6. `main.py` Trains POMSL model.
7. `utils.py` Metric calculation.
8. `model.py` The main components of the POMSL.
10. `M_D.csv` Metabolite-Disease Adjacency Matrix.
        

