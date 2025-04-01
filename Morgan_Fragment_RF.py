import os
import pandas as pd
import chardet 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from boruta import BorutaPy

# Set the directory
path = "/Users/ac3124/Desktop/"
os.chdir(path)

# 1 SETUP
# Load the files
meta = pd.read_csv("meta.csv") 
feat = pd.read_csv("feat.csv")
morgan = pd.read_csv("morgan.csv")   # Morgan fingerprint file

# Detect encoding of Drug_SMILES.csv
with open("Drug_SMILES.csv", "rb") as f:
    result = chardet.detect(f.read(100000))  # Read first 100KB for detection

detected_encoding = result["encoding"]
print(f"📢 Detected Encoding: {detected_encoding}")

# Load the second CSV file with detected encoding
try:
    drug_smiles_df = pd.read_csv("Drug_SMILES.csv", encoding=detected_encoding)
except UnicodeDecodeError:
    print("⚠️ Encoding error detected! Trying 'ISO-8859-1' instead...")
    drug_smiles_df = pd.read_csv("Drug_SMILES.csv", encoding="ISO-8859-1")

# Display first few rows to inspect structure
display(meta.head())
display(drug_smiles_df.head())

# Ensure correct column names
meta_column = "drug_type"  # Column in meta.csv with drug names
smiles_column = "Compound"  # Column in Drug_SMILES.csv with compounds

# Remove NaN values before processing
meta = meta.dropna(subset=[meta_column])
drug_smiles_df = drug_smiles_df.dropna(subset=[smiles_column])

# 2 NAME MATCHING
# Convert to lowercase and remove extra spaces for comparison
meta_drugs = meta[meta_column].astype(str).str.lower().str.strip().tolist()
drug_smiles = drug_smiles_df[smiles_column].astype(str).str.lower().str.strip().tolist()

# Find matching and non-matching compounds
matches = [drug for drug in drug_smiles if drug in meta_drugs]
non_matches = [drug for drug in drug_smiles if drug not in meta_drugs]

# Count the number of matches and non-matches
num_matches = len(matches)
num_non_matches = len(non_matches)

# Ensure both lists have the same length by padding the shorter one
max_length = max(len(matches), len(non_matches))
matches += [None] * (max_length - len(matches))
non_matches += [None] * (max_length - len(non_matches))

# Create a DataFrame
result_df = pd.DataFrame({"Matching Drugs": matches, "Non-Matching Drugs": non_matches})

# Save results to a CSV file
result_df.to_csv("drug_matching_results.csv", index=False)
print("✅ Results saved as 'drug_matching_results.csv'")

# Print final output
print(f"✅ Total Matching Drugs: {num_matches}")
print("✅ Matching Drugs:", matches)
print(f"❌ Total Non-Matching Drugs: {num_non_matches}")
print("❌ Non-Matching Drugs:", non_matches)

# 3 DATA EXTRACTION
#Extract rows that contain "bluelight" in the first column of meta.csv
meta_bl = meta[meta.iloc[:, 0].str.contains("bluelight", case=False, na=False)]

# Merge meta_bl with morgan.csv on 'drug_type'
meta_bl_morgan = pd.merge(meta_bl, morgan, on="drug_type", how="inner")

# Extract feat.csv rows where 'well_id' matches meta_bl_morgan
feat_bl = feat[feat["well_id"].isin(meta_bl_morgan["well_id"])]

# Merge `drug_type` from `meta_bl_morgan` into `feat_bl` based on `well_id`
feat_bl = feat_bl.merge(meta_bl_morgan[['well_id', 'drug_type']], on='well_id', how='left')

# Save extracted data to new CSV files
meta_bl.to_csv("meta_bl.csv", index=False)
meta_bl_morgan.to_csv("meta_bl_morgan.csv", index=False)
feat_bl.to_csv("feat_bl.csv", index=False)  # Now includes 'drug_type'

# Print completion message
print("✅ Process completed! Extracted files saved as 'meta_bl.csv', 'meta_bl_morgan.csv', and 'feat_bl.csv'.")

# 4 FEATURE SELECTION AND MERGE IN NEW DATASET
# Define the feature set from feat_bl.csv
tierpsy_16 = [
    'length_90th', 'width_midbody_norm_10th', 'curvature_hips_abs_90th', 'curvature_head_abs_90th',
    'motion_mode_paused_fraction', 'd_curvature_head_abs_90th', 'width_head_base_norm_10th',
    'quirkiness_50th', 'minor_axis_50th', 'curvature_midbody_norm_abs_50th',
    'motion_mode_paused_frequency', 'd_curvature_hips_abs_90th', 'motion_mode_backward_frequency',
    'relative_to_hips_radial_velocity_tail_tip_50th', 'relative_to_head_base_radial_velocity_head_tip_50th',
    'relative_to_head_base_angular_velocity_head_tip_abs_90th'
]

# Keep only `tierpsy_16` features and `drug_type` from feat_bl.csv
feat_bl = feat_bl[["drug_type"] + tierpsy_16]

# Merge feat_bl with morgan.csv on 'drug_type'
merged_data = pd.merge(feat_bl, morgan, on="drug_type", how="inner")

# 5 RANDOM FOREST
# Identify Morgan fragment columns (0-2047)
morgan_features = [str(i) for i in range(204)]  # Morgan fingerprints are named as '0' to '2047'
existing_morgan_features = [col for col in morgan_features if col in merged_data.columns]  # Ensure they exist

# Store important features
important_fragments = set()

# Store important fragments across all features
important_fragments = set()

# Define parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20],
    'min_samples_leaf': [1, 5],
    'max_features': ['sqrt', 'log2']
}

# Run Random Forest with Grid Search for each feature
for feature in tierpsy_16:
    print(f"\n🔍 Running Grid Search for: {feature}")
    
    X = merged_data[existing_morgan_features]
    y = merged_data[feature]
    
    if y.isna().sum() > 0:
        print(f"⚠️ Found {y.isna().sum()} missing values. Removing them...")
        X = X[~y.isna()]
        y = y.dropna()

    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Grid Search with 3-fold CV
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        rf_base, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    rf = grid_search.best_estimator_
    print(f"✅ Best parameters for {feature}: {grid_search.best_params_}")
    
    # Evaluate performance
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"📊 MSE for {feature}: {mse:.4f}")

    # Extract feature importances
    feature_importances = pd.DataFrame({
        "Feature": existing_morgan_features,
        "Importance": rf.feature_importances_
    })
    #feature_importances = feature_importances[feature_importances["Importance"] > 0.01]
    # Sort and filter top 20 features
    plot_df = feature_importances.sort_values(by="Importance", ascending=False).head(20).copy()

    # Assign pastel colors based on importance thresholds
    def assign_color(val):
        if val >= 0.01:
            return "#ff9999"  # pastel red
        elif 0.005 <= val < 0.01:
            return "#99ccff"  # pastel blue (complementary)
        else:
            return "#cccccc"  # grey

    colors = [assign_color(val) for val in plot_df["Importance"]]

    # Plot with custom color scheme
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=plot_df,
        palette=colors
    )
    plt.title(f"Feature Importance for {feature}")
    plt.xlabel("Importance")
    plt.ylabel("Morgan Fragment (Bit Index)")
    plt.tight_layout()
    plt.show()