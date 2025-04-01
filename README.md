
# Performs breakdown of SMILES to Morgan fragments and using Random forest to identify important fragments contributing to the specific Tierpsy features
# Run the codes in following file order

# 1. SMILES_Analysis.py
# Calculates Tanimoto similarity, gives a dendrogram of hierarchal clusters based on distance of similarity and a CSV summary file

# 2. Morgan_Fragment_RF.py
# Doing RF regressor
# Work in progress: noise filtering for RF (either using specific morgan fragment radius or covariate analysis for filtering)
