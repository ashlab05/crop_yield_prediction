"""
Data Preprocessing Module
=========================
Loads yield_df.csv, encodes categoricals, scales features, and applies
standard 80/20 train/test split.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch


def load_and_preprocess(csv_path="data/yield_df.csv", test_size=0.2, random_state=42):
    """
    Load, encode, scale, and split the crop yield dataset.
    
    Returns:
        dict with keys:
            X_train, X_test, y_train, y_test  (numpy arrays)
            feature_names, scaler, le_area, le_item
            df (full DataFrame with encoded columns)
            num_countries, num_crops
    """
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    # Drop rows with missing yield
    df = df.dropna(subset=['hg/ha_yield'])
    
    # Encode categoricals
    le_area = LabelEncoder()
    le_item = LabelEncoder()
    df['country_id'] = le_area.fit_transform(df['Area'])
    df['crop_id'] = le_item.fit_transform(df['Item'])
    
    num_countries = df['country_id'].nunique()
    num_crops = df['crop_id'].nunique()
    
    # Feature columns (numeric)
    numeric_features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
    
    # Full feature matrix: country_id, crop_id, Year, + numeric
    feature_cols = ['country_id', 'crop_id', 'Year'] + numeric_features
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['hg/ha_yield'].values.astype(np.float32)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"  Dataset: {len(df)} samples")
    print(f"  Countries: {num_countries}, Crops: {num_crops}")
    print(f"  Features: {feature_cols}")
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Yield range: [{y.min():.0f}, {y.max():.0f}]")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'le_area': le_area, 'le_item': le_item,
        'df': df,
        'num_countries': num_countries,
        'num_crops': num_crops,
    }


def to_torch(data, device='cpu'):
    """Convert numpy data dict to torch tensors."""
    return {
        'X_train': torch.tensor(data['X_train'], dtype=torch.float32).to(device),
        'X_test': torch.tensor(data['X_test'], dtype=torch.float32).to(device),
        'y_train': torch.tensor(data['y_train'], dtype=torch.float32).to(device),
        'y_test': torch.tensor(data['y_test'], dtype=torch.float32).to(device),
    }
