#!/usr/bin/env python3
"""
Crop Yield Prediction - GNN and DANN Implementation
====================================================
This script implements:
1. Climate-Similarity Graph Neural Networks (CR-GNN)
2. Domain Adversarial Neural Networks (DANN)

Run this file directly in VSCode/terminal instead of Jupyter Notebook.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

print("=" * 80)
print("Crop Yield Prediction - GNN and DANN Implementation")
print("=" * 80)

# =============================================================================
# Step 1: Load Data
# =============================================================================
print("\n📊 Loading data...")
df = pd.read_csv("yield_df.csv")

# Drop unnamed column if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Unique countries: {df['Area'].nunique()}")
print(f"Unique crops: {df['Item'].nunique()}")

# =============================================================================
# Step 2: Data Preprocessing
# =============================================================================
print("\n🔧 Preprocessing data...")

# Encode countries and items
le_area = LabelEncoder()
le_item = LabelEncoder()

df['country_id'] = le_area.fit_transform(df['Area'])
df['item_id'] = le_item.fit_transform(df['Item'])

# Get unique countries
countries = sorted(df['Area'].unique())
country_to_idx = {c: i for i, c in enumerate(countries)}
num_countries = len(countries)

print(f"Number of countries: {num_countries}")

# =============================================================================
# Step 3: Define Custom GNN Layer
# =============================================================================
class GraphConvolution(layers.Layer):
    """Custom Graph Convolution Layer for GNN."""
    
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel', 
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer='zeros',
                name='bias', 
                trainable=True
            )

    def call(self, inputs):
        features, adj = inputs
        # features: (Batch, Nodes, Feats)
        # adj: (Nodes, Nodes)
        
        # Support = Features * Kernel
        support = tf.matmul(features, self.kernel)
        
        # Output = Adjacency * Support
        output = tf.matmul(adj, support)
        
        if self.use_bias:
            output = output + self.bias
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output

print("✓ Custom GraphConvolution Layer defined.")

# =============================================================================
# Step 4: Construct Climate-Similarity Adjacency Matrix
# =============================================================================
print("\n🌍 Constructing Climate-Similarity Graph...")

# Calculate average climate vector for each country
climate_cols = ['average_rain_fall_mm_per_year', 'avg_temp']
country_climate_vectors = df.groupby('Area')[climate_cols].mean().reindex(countries).fillna(0).values

# Normalize vectors
scaler_clim = StandardScaler()
country_climate_vectors_scaled = scaler_clim.fit_transform(country_climate_vectors)

# Compute Cosine Similarity
sim_matrix = cosine_similarity(country_climate_vectors_scaled)

# Threshold to create a graph
threshold = 0.95
adj_climate = (sim_matrix > threshold).astype(float)
np.fill_diagonal(adj_climate, 1.0)  # Self-loops

# Geographic adjacency (Identity as fallback)
adj_geo = np.eye(num_countries)

def normalize_adj(adj):
    """Normalize adjacency matrix (D^-0.5 A D^-0.5)."""
    d = np.diag(np.sum(adj, axis=1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    return d_inv_sqrt.dot(adj).dot(d_inv_sqrt)

adj_climate_norm = normalize_adj(adj_climate)
adj_geo_norm = normalize_adj(adj_geo)

print(f"✓ Climate Graph: {int(np.sum(adj_climate))} edges for {num_countries} nodes.")

# =============================================================================
# Step 5: Prepare Spatial Split (Leave-Countries-Out)
# =============================================================================
print("\n📌 Creating Spatial Split (Leave-Countries-Out)...")

np.random.seed(42)
tf.random.set_seed(42)

all_countries = df['country_id'].unique()
n_val = int(len(all_countries) * 0.2)
test_countries = np.random.choice(all_countries, size=n_val, replace=False)
train_countries = np.setdiff1d(all_countries, test_countries)

print(f"  Train Countries: {len(train_countries)}")
print(f"  Test Countries (Unseen): {len(test_countries)}")

# Create masks
train_mask = df['country_id'].isin(train_countries)
test_mask = df['country_id'].isin(test_countries)

# Prepare feature arrays
feature_cols = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
scaler_f = StandardScaler()

X_c = df['country_id'].values
X_f = scaler_f.fit_transform(df[feature_cols].values)
y = df['hg/ha_yield'].values

X_c_train, X_f_train = X_c[train_mask], X_f[train_mask]
y_train = y[train_mask]

X_c_test, X_f_test = X_c[test_mask], X_f[test_mask]
y_test = y[test_mask]

print(f"  Train samples: {len(y_train)}")
print(f"  Test samples: {len(y_test)}")

# =============================================================================
# Step 6: Build and Train GNN Models
# =============================================================================
def build_gnn_model_spatial(adj_tensor, num_countries):
    """Build a GNN model for spatial generalization."""
    country_input = layers.Input(shape=(1,), dtype='int32', name='country_idx')
    other_input = layers.Input(shape=(3,), name='other_feats')
    
    t_adj = tf.constant(adj_tensor, dtype=tf.float32)
    
    # Node embeddings
    node_embeddings = layers.Embedding(num_countries, 32)(tf.range(num_countries))
    x_graph = tf.expand_dims(node_embeddings, 0)  # (1, Nodes, 32)
    
    # GCN Layers
    x_graph = GraphConvolution(32, activation='relu')([x_graph, t_adj])
    x_graph = GraphConvolution(16, activation='relu')([x_graph, t_adj])
    
    # Get embedding for specific country
    node_vecs = x_graph[0]
    specific_country_vec = tf.gather(node_vecs, country_input)
    specific_country_vec = layers.Flatten()(specific_country_vec)
    
    # Prediction head
    concat = layers.Concatenate()([specific_country_vec, other_input])
    x = layers.Dense(64, activation='relu')(concat)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1)(x)
    
    model = models.Model(inputs=[country_input, other_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

print("\n🧠 Training Climate-Similarity GNN...")
model_c_gnn = build_gnn_model_spatial(adj_climate_norm, num_countries)
hist_c = model_c_gnn.fit(
    [X_c_train, X_f_train], y_train,
    validation_data=([X_c_test, X_f_test], y_test),
    epochs=15, batch_size=64, verbose=1
)

print("\n🧠 Training Baseline Geographic GNN...")
model_g_gnn = build_gnn_model_spatial(adj_geo_norm, num_countries)
hist_g = model_g_gnn.fit(
    [X_c_train, X_f_train], y_train,
    validation_data=([X_c_test, X_f_test], y_test),
    epochs=15, batch_size=64, verbose=0
)

print("\n🧠 Training Baseline MLP (Country-Blind)...")
input_mlp = layers.Input(shape=(3,))
x = layers.Dense(128, activation='relu')(input_mlp)
x = layers.Dense(64, activation='relu')(x)
out = layers.Dense(1)(x)
model_mlp = models.Model(input_mlp, out)
model_mlp.compile(loss='mse', metrics=['mae'], optimizer='adam')
model_mlp.fit(X_f_train, y_train, validation_data=(X_f_test, y_test), epochs=15, verbose=0)

print("\n" + "=" * 60)
print("📊 GNN SPATIAL GENERALIZATION RESULTS (MAE)")
print("=" * 60)
mae_c = model_c_gnn.evaluate([X_c_test, X_f_test], y_test, verbose=0)[1]
mae_g = model_g_gnn.evaluate([X_c_test, X_f_test], y_test, verbose=0)[1]
mae_mlp = model_mlp.evaluate(X_f_test, y_test, verbose=0)[1]

print(f"1. Baseline MLP (No Country info):   MAE = {mae_mlp:.0f}")
print(f"2. Geographic GNN (Identity):        MAE = {mae_g:.0f}")
print(f"3. Climate-Similarity GNN (Ours):    MAE = {mae_c:.0f}")

if mae_c < mae_g:
    print("\n✅ SUCCESS: Climate GNN generalizes better than Geographic baseline!")
else:
    print("\n📈 Note: Climate GNN performance is comparable - may need tuning.")

# =============================================================================
# Step 7: Domain Adversarial Neural Network (DANN)
# =============================================================================
print("\n" + "=" * 80)
print("🎯 DOMAIN ADVERSARIAL TRAINING (DANN)")
print("=" * 80)

@tf.custom_gradient
def gradient_reversal(x):
    """Gradient Reversal Layer - reverses gradients during backprop."""
    def grad(dy):
        return -1.0 * dy
    return x, grad

class GradientReversalLayer(layers.Layer):
    """Keras layer that reverses gradients during backpropagation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        return gradient_reversal(x)

def build_dann_model(input_shape, num_countries):
    """Build Domain Adversarial Neural Network."""
    inputs = layers.Input(shape=input_shape)
    
    # Shared Feature Extractor
    x = layers.Dense(128, activation='relu')(inputs)
    features = layers.Dense(64, activation='relu')(x)
    
    # 1. Yield Predictor (Main Task)
    y_branch = layers.Dense(32, activation='relu')(features)
    yield_output = layers.Dense(1, name='yield_output')(y_branch)
    
    # 2. Country Classifier (Adversarial Task with Gradient Reversal)
    grl_features = GradientReversalLayer()(features)
    c_branch = layers.Dense(32, activation='relu')(grl_features)
    country_output = layers.Dense(num_countries, activation='softmax', name='country_output')(c_branch)
    
    model = models.Model(inputs=inputs, outputs=[yield_output, country_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'yield_output': 'mse', 
            'country_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'yield_output': 1.0, 
            'country_output': 0.1
        }
    )
    return model

print("\n🧠 Training Domain Adversarial Network (DANN)...")
dann_model = build_dann_model((3,), num_countries)

history_dann = dann_model.fit(
    X_f_train, 
    {'yield_output': y_train, 'country_output': X_c_train},
    validation_data=(X_f_test, {'yield_output': y_test, 'country_output': X_c_test}),
    epochs=15, batch_size=64, verbose=1
)

# Evaluate DANN
results = dann_model.evaluate(X_f_test, {'yield_output': y_test, 'country_output': X_c_test}, verbose=0)
dann_mse = results[1]  # yield_output loss (MSE)
dann_rmse = np.sqrt(dann_mse)
dann_country_acc = results[2]  # country_output accuracy

print("\n" + "=" * 60)
print("📊 DANN RESULTS")
print("=" * 60)
print(f"Yield RMSE: {dann_rmse:.2f}")
print(f"Country Classification Loss: {results[2]:.4f}")
print("\n💡 Lower country classification = features are more country-invariant")

# =============================================================================
# Step 8: Summary Comparison
# =============================================================================
print("\n" + "=" * 80)
print("📋 FINAL SUMMARY - SPATIAL GENERALIZATION")
print("=" * 80)
print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    SPATIAL GENERALIZATION RESULTS                    │
├─────────────────────────────────────────────────────────────────────┤
│  Model                           │  MAE on Unseen Countries         │
├──────────────────────────────────┼──────────────────────────────────┤
│  MLP (No Country Info)           │  {:.0f}                          
│  Geographic GNN (Identity)       │  {:.0f}                          
│  Climate-Similarity GNN (Ours)   │  {:.0f}                          
│  DANN (Country-Invariant)        │  RMSE = {:.0f}                   
└─────────────────────────────────────────────────────────────────────┘
""".format(mae_mlp, mae_g, mae_c, dann_rmse))

print("""
🎯 KEY INSIGHTS:
   1. Climate-Similarity GNN explicitly models climate relationships
   2. DANN forces features to be country-invariant via gradient reversal
   3. Both approaches aim to improve generalization to unseen countries

📚 This implements the novel approaches from the research paper:
   - "Generalizing Crop Yield Prediction: A Climate-Regime Aware Approach"
""")

print("\n✅ Script completed successfully!")
print("=" * 80)
