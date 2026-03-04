#!/usr/bin/env python3
"""
Script 05: DANN Model Definition
Defines Domain Adversarial Neural Network with Gradient Reversal.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Step 5: DANN Model Definition")
print("=" * 60)

# Define Gradient Reversal Layer
@tf.custom_gradient
def gradient_reversal_fn(x):
    """Gradient reversal function - forward pass is identity, backward reverses."""
    def grad(dy):
        return -1.0 * dy
    return x, grad


class GradientReversalLayer(layers.Layer):
    """
    Gradient Reversal Layer for Domain Adversarial Training.
    Forward pass: identity
    Backward pass: reverses gradients (multiplies by -1)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        return gradient_reversal_fn(x)
    
    def get_config(self):
        return super().get_config()


print("✓ GradientReversalLayer defined")

def build_dann_model(input_shape, num_countries):
    """
    Build Domain Adversarial Neural Network.
    
    Architecture:
    1. Shared Feature Extractor
    2. Yield Predictor (main task)
    3. Country Classifier with Gradient Reversal (adversarial task)
    
    The gradient reversal forces the feature extractor to learn
    country-invariant representations that still predict yield well.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Shared Feature Extractor
    x = layers.Dense(128, activation='relu', name='fe_dense1')(inputs)
    x = layers.Dropout(0.2)(x)
    features = layers.Dense(64, activation='relu', name='fe_dense2')(x)
    
    # Branch 1: Yield Predictor (Main Task)
    y_branch = layers.Dense(32, activation='relu', name='yield_dense')(features)
    yield_output = layers.Dense(1, name='yield_output')(y_branch)
    
    # Branch 2: Country Classifier (Adversarial Task with GRL)
    grl_features = GradientReversalLayer(name='gradient_reversal')(features)
    c_branch = layers.Dense(32, activation='relu', name='country_dense')(grl_features)
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
            'country_output': 0.1  # Lower weight for adversarial loss
        },
        metrics={
            'yield_output': ['mae'],
            'country_output': ['accuracy']
        }
    )
    
    return model


# Test model creation
print("\n🧪 Testing DANN model creation...")
data = np.load('scripts/preprocessed_data.npz')
num_countries = int(data['num_countries'])

test_model = build_dann_model((3,), num_countries)
print(f"✓ DANN model created successfully")
print(f"  • Input shape: {test_model.input.shape}")
print(f"  • Output shapes: {[o.shape for o in test_model.outputs]}")
print(f"  • Total params: {test_model.count_params():,}")

# Quick forward pass test
X_test_sample = np.random.randn(3, 3).astype(np.float32)
pred = test_model.predict(X_test_sample, verbose=0)
print(f"  • Yield prediction shape: {pred[0].shape}")
print(f"  • Country prediction shape: {pred[1].shape}")

print("\n✅ DANN model definition complete!")
