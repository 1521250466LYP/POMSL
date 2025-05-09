# POMSL：Progressive Orthogonal Multimodal Similarity Learning for Metabolite-Disease Association Prediction


##### Requirement


- numpy                     1.23.5
- pandas                    2.2.2
- python                    3.9.19
- scipy                     1.13.1
- torch                     2.2.1+cu121              
- scikit-learn              1.5.1


##### Dataset

- **Statistics**
    - Metabolites：2262
    - Diseases: 216
    - Validated metabolite-disease assiciations: 4536

##### File Introduction:

1. `ConstructHW.py` Construct hypergraphs.
2. `getsample.py` Obtain the positive and negative sample sets.
3. `getsimilarity.py` Obtain multiple similarities.
4. `hypergraph_construct_KNN.py` Construct a hypergraph using KNN.
5. `hypergraph_construct_kmeans.py` Construct a hypergraph using K-means.
6. `main.py` Trains POMSL model.
7. `utils.py` Metric calculation.
8. `model.py` The main components of the POMSL.
10. `M_D.zip` Metabolite-Disease Adjacency Matrix.
11. `metabolite_GIP_similarity.zip` Metabolite Gaussian Interaction Profile Kernel Similarity(MGIP).
12. `metabolites_information_entropy_similarity.zip` Metabolite Similarity based on Information Entropy (MSIE).
13. `metabolites_structure_similarity.zip` Metabolite Structural Similarity (MSS).
14. `disease_GIP_similarity.csv` Disease Gaussian Interaction Profile Kernel Similarity (DGIP).
15. `disease _information_entropy_similarity .csv` Disease Similarity based on Information Entropy (DSIE).
16. `disease_semantic_similarity.csv` Disease Semantic Similarity (DSS).
17. `disease name.xlsx` Disease Names Included in the Selected Dataset.
18. `metabolite name.xlsx` Metabolite Names Included in the Selected Dataset.

##### Running for POMSL

1. Extract all .zip files
2. Set up the environment based on the provided configuration.
3. Run the `main.py` file to start the model training and prediction.


        

