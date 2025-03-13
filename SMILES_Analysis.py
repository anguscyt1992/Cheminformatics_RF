import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw

from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

# 1 SETUP
# Disable RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

# Load data
df = pd.read_csv('/Users/ac3124/Desktop/Drug_SMILES.csv', encoding='latin-1')
df = df.dropna(subset=['SMILES'])
df['SMILES'] = df['SMILES'].astype(str).str.strip()
df['Compound'] = df['Compound'].astype(str).str.strip()
compounds = df['Compound'].tolist()
smiles_list = df['SMILES'].tolist()

# Function to generate Morgan fingerprint with bit information
def get_fp_and_bitinfo(mol, radius=2, nBits=2048):
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=bit_info)
    return fp, bit_info

# Generate fingerprints for valid compounds
mols = []
valid_compounds = []
fps = []
bitinfo_list = []
for comp, smi in zip(compounds, smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp, bit_info = get_fp_and_bitinfo(mol, radius=2, nBits=2048)
        mols.append(mol)
        fps.append(fp)
        bitinfo_list.append(bit_info)
        valid_compounds.append(comp)
    else:
        print(f"Warning: Invalid SMILES skipped: {smi}")
n = len(fps)
if n == 0:
    raise ValueError("No valid SMILES were found. Exiting.")


# Compute Tanimoto similarity matrix and derive distance matrix
similarity_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        similarity_matrix[i, j] = sim
        similarity_matrix[j, i] = sim
distance_matrix = 1 - similarity_matrix
condensed_distance = squareform(distance_matrix)
# Perform clustering using average linkage
linkage_matrix = linkage(condensed_distance, method='average')


# 2 CLUSTERING
# Cluster compounds using a distance threshold (e.g., 0.3 → Tanimoto similarity ≈ 0.7)
cluster_threshold = 0.5

# Assign cluster labels using the same threshold.
cluster_labels = fcluster(linkage_matrix, t=cluster_threshold, criterion='distance')
clusters_df = pd.DataFrame({'Compound': valid_compounds, 'Cluster': cluster_labels})
print("Cluster assignments:")
print(clusters_df.sort_values('Cluster'))

# Filter to keep only clusters with more than one compound
cluster_sizes = clusters_df['Cluster'].value_counts()
valid_clusters = cluster_sizes[cluster_sizes > 1].index.tolist()
clusters_df_filtered = clusters_df[clusters_df['Cluster'].isin(valid_clusters)]
print("\nFiltered cluster assignments (clusters with >1 compound):")
print(clusters_df_filtered.sort_values('Cluster'))

# 3 ENRICHMENT ANALYSIS
def compute_bit_frequencies(fp_list, nBits=2048):
    counts = np.zeros(nBits, dtype=float)
    for fp in fp_list:
        for b in fp.GetOnBits():
            counts[b] += 1
    return counts / len(fp_list)

overall_freq = compute_bit_frequencies(fps, nBits=2048)
cluster_bit_freq = {}
unique_clusters = np.unique(cluster_labels)
for c in unique_clusters:
    fps_cluster = [fp for fp, cl in zip(fps, cluster_labels) if cl == c]
    cluster_bit_freq[c] = compute_bit_frequencies(fps_cluster, nBits=2048) if fps_cluster else np.zeros(2048)

cluster_enrichment = {}
for c in unique_clusters:
    enrichment = cluster_bit_freq[c] - overall_freq
    enriched_bits = [i for i, val in enumerate(enrichment) if val > 0]
    # Take top 5 enriched bits (sorted highest first)
    enriched_bits_sorted = sorted(enriched_bits, key=lambda x: enrichment[x], reverse=True)
    cluster_enrichment[c] = enriched_bits_sorted[:5]

for c in unique_clusters:
    print(f"\nCluster {c}: Top enriched fingerprint bits: {cluster_enrichment[c]}")

# 4 HIERARCHICAL CLUSTERING DENDROGRAM
plt.figure(figsize=(20, 10), dpi=300)
ddata = dendrogram(linkage_matrix,
                   labels=valid_compounds,
                   leaf_rotation=90,
                   color_threshold=cluster_threshold)
plt.title("Hierarchical Clustering Dendrogram (Threshold = 0.7)")
plt.xlabel("Compound")
plt.ylabel("Distance (1 - Tanimoto Similarity)")
plt.tight_layout()
# Save high-res image
plt.savefig("dendrogram_highres.png", dpi=300)
plt.show()

# 5 CLUSTERING CSV OUTPUT
clusters_df = pd.DataFrame({'Compound': valid_compounds, 'Cluster': cluster_labels})
clusters_df_sorted = clusters_df.sort_values('Cluster')
print("Cluster assignments:")
print(clusters_df_sorted)

output_csv = "/Users/ac3124/Desktop/cluster_assignments.csv"  # change path as needed
clusters_df_sorted.to_csv(output_csv, index=False)
print(f"Cluster assignments exported to '{output_csv}'.")

# CLUSTER BIT VISUALIZATION
def highlight_bit(mol, bit_id, bit_info, img_size=(300,300)):
    if bit_id not in bit_info:
        return Draw.MolToImage(mol, size=img_size)
    atoms_to_highlight = {atom_idx for atom_idx, radius in bit_info[bit_id]}
    return Draw.MolToImage(mol, size=img_size, highlightAtoms=list(atoms_to_highlight))

output_dir = "/Users/ac3124/Desktop/cluster_substructures"
os.makedirs(output_dir, exist_ok=True)

# Loop over each valid cluster (clusters with >1 compound)
for c in valid_clusters:
    # Get indices of compounds in the current cluster
    indices = [i for i, cl in enumerate(cluster_labels) if cl == c]
    print(f"\nCluster {c} has {len(indices)} compounds: {[valid_compounds[i] for i in indices]}")
    
    enriched_bits = cluster_enrichment[c]
    n_rows = len(enriched_bits)
    n_cols = len(indices)
    
    # Create a grid: rows = enriched bits, columns = compounds in the cluster
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    # Ensure axs is a 2D array
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)
    
    for r, bit in enumerate(enriched_bits):
        for c_idx, i in enumerate(indices):
            mol = mols[i]
            comp_name = valid_compounds[i]
            bit_info = bitinfo_list[i]
            # Highlight the substructure corresponding to the enriched bit if available.
            if bit in bit_info:
                img = highlight_bit(mol, bit, bit_info, img_size=(300,300))
            else:
                img = Draw.MolToImage(mol, size=(300,300))
            axs[r, c_idx].imshow(img)
            axs[r, c_idx].axis('off')
            # Only add compound names as column titles on the first row.
            if r == 0:
                axs[r, c_idx].set_title(comp_name, fontsize=8)
        # Label each row with the enriched bit (on the left side)
        axs[r, 0].set_ylabel(f"Bit {bit}", fontsize=10)
    
    plt.suptitle(f"Cluster {c}: Enriched Bits", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = os.path.join(output_dir, f"cluster_{c}_all_bits.png")
    print(f"Saved figure to {filename}")
    plt.show()