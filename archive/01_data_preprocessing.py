#!/usr/bin/env python3
"""
Script 01: Data Preprocessing
Loads and preprocesses the crop yield data.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Change to project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Step 1: Data Preprocessing")
print("=" * 60)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv("yield_df.csv")

# Drop unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Columns: {list(df.columns)}")
print(f"✓ Unique countries: {df['Area'].nunique()}")
print(f"✓ Unique crops: {df['Item'].nunique()}")

# Encode countries and items
print("\n🔧 Encoding categorical variables...")
le_area = LabelEncoder()
le_item = LabelEncoder()

df['country_id'] = le_area.fit_transform(df['Area'])
df['item_id'] = le_item.fit_transform(df['Item'])

# Get unique countries
countries = sorted(df['Area'].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
num_countries = len(countries)

print(f"✓ Encoded {num_countries} countries")
print(f"✓ Encoded {df['Item'].nunique()} crop types")

# Create spatial split (Leave-Countries-Out)
print("\n📌 Creating Spatial Split (Leave-Countries-Out)...")
np.random.seed(42)

all_countries = df['country_id'].unique()
n_val = int(len(all_countries) * 0.2)
test_countries = np.random.choice(all_countries, size=n_val, replace=False)
train_countries = np.setdiff1d(all_countries, test_countries)

print(f"✓ Train Countries: {len(train_countries)}")
print(f"✓ Test Countries (Unseen): {len(test_countries)}")

# Create masks
train_mask = df['country_id'].isin(train_countries)
test_mask = df['country_id'].isin(test_countries)

# Prepare feature arrays
feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
scaler_f = StandardScaler()

X_c = df['country_id'].values
X_f_all = df[feature_cols].values
X_f = scaler_f.fit_transform(X_f_all)
y = df['hg/ha_yield'].values

X_c_train, X_f_train = X_c[train_mask], X_f[train_mask]
y_train = y[train_mask]

X_c_test, X_f_test = X_c[test_mask], X_f[test_mask]
y_test = y[test_mask]

print(f"✓ Train samples: {len(y_train)}")
print(f"✓ Test samples: {len(y_test)}")

# Save processed data for other scripts
print("\n💾 Saving preprocessed data...")
np.savez('scripts/preprocessed_data.npz',
         X_c_train=X_c_train, X_f_train=X_f_train, y_train=y_train,
         X_c_test=X_c_test, X_f_test=X_f_test, y_test=y_test,
         num_countries=num_countries)

# Save country climate vectors for graph construction
climate_cols = ['average_rain_fall_mm_per_year', 'avg_temp']
country_climate_vectors = df.groupby('Area')[climate_cols].mean().reindex(countries).fillna(0).values
np.save('scripts/country_climate_vectors.npy', country_climate_vectors)

print("✓ Saved preprocessed_data.npz")
print("✓ Saved country_climate_vectors.npy")
print("\n✅ Data preprocessing complete!")
