"""
tms_model.py
SignBridge ML — TMS-Attention Model Architecture

Architecture:
  Input: (batch, 64, 126) — sequence of normalized landmark vectors

  Branch A (local spatial): 1D Conv layers (MobileNet-style depthwise separable)
    Kernel sizes: 3, 5, 7. Output: (batch, 64, 128)

  Branch B (temporal attention): Multi-Head Self-Attention (4 heads, d_model=128)
    Positional encoding added. Output: (batch, 64, 128)

  Conformer block between Conv output and Attention
    Conv module: Pointwise → Depthwise Conv → GLU → Pointwise
    Self-attention module: Multi-head attention with relative positional encoding
    Feed-forward module: Two linear layers with SiLU activation

  Concatenate: (batch, 64, 256) → Global Average Pooling → Dense(128, ReLU)
    → Dropout(0.3) → Dense(N_CLASSES, Softmax)

  CTC decode layer: wraps model in CTC framework for continuous signing.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


# ============================================================
# Positional Encoding
# ============================================================

@keras.utils.register_keras_serializable(package='SignBridge')
class PositionalEncoding(layers.Layer):
    """Sinusoidal positional encoding for transformer-style attention."""
    
    def __init__(self, max_len=128, d_model=128, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Precompute positional encoding matrix
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, :, :])  # (1, max_len, d_model)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config


# ============================================================
# Branch A: MobileNet-style Depthwise Separable Conv
# ============================================================

@keras.utils.register_keras_serializable(package='SignBridge')
class DepthwiseSeparableConv1D(layers.Layer):
    """MobileNet-style depthwise separable 1D convolution."""
    
    def __init__(self, filters, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self.depthwise = layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding='same', activation=None
        )
        self.pointwise = layers.Conv1D(filters, 1, activation=None)
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()
    
    def call(self, x, training=False):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'filters': self._filters, 'kernel_size': self._kernel_size})
        return config


def build_conv_branch(input_shape, filters=128):
    """
    Branch A: Multi-scale 1D depthwise separable convolutions.
    Kernel sizes: 3, 5, 7 — captures local spatial patterns at different scales.
    
    Input:  (batch, seq_len, features)
    Output: (batch, seq_len, filters)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Project to working dimension
    x = layers.Dense(filters)(inputs)
    
    # Multi-scale convolutions
    conv3 = DepthwiseSeparableConv1D(filters // 3, kernel_size=3)(x)
    conv5 = DepthwiseSeparableConv1D(filters // 3, kernel_size=5)(x)
    conv7 = DepthwiseSeparableConv1D(filters - 2 * (filters // 3), kernel_size=7)(x)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()([conv3, conv5, conv7])
    
    # Additional refinement
    x = DepthwiseSeparableConv1D(filters, kernel_size=3)(x)
    
    return Model(inputs, x, name='conv_branch')


# ============================================================
# Conformer Block (Step 2.3)
# ============================================================

@keras.utils.register_keras_serializable(package='SignBridge')
class ConformerFeedForward(layers.Layer):
    """Feed-forward module: two linear layers with SiLU activation."""
    
    def __init__(self, d_model=128, expansion_factor=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model
        self._expansion_factor = expansion_factor
        self._dropout_rate = dropout_rate
        self.ln = layers.LayerNormalization()
        self.dense1 = layers.Dense(d_model * expansion_factor)
        self.dense2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        residual = x
        x = self.ln(x)
        x = self.dense1(x)
        x = tf.nn.silu(x)  # SiLU activation
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return residual + 0.5 * x
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self._d_model, 'expansion_factor': self._expansion_factor, 'dropout_rate': self._dropout_rate})
        return config


@keras.utils.register_keras_serializable(package='SignBridge')
class ConformerConvModule(layers.Layer):
    """
    Conformer conv module:
    Pointwise → Depthwise Conv → GLU → Pointwise
    """
    
    def __init__(self, d_model=128, kernel_size=31, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model
        self._kernel_size = kernel_size
        self._dropout_rate = dropout_rate
        self.ln = layers.LayerNormalization()
        self.pointwise1 = layers.Dense(2 * d_model)
        self.depthwise = layers.DepthwiseConv1D(
            kernel_size=kernel_size, padding='same'
        )
        self.bn = layers.BatchNormalization()
        self.pointwise2 = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, x, training=False):
        residual = x
        x = self.ln(x)
        x = self.pointwise1(x)
        
        # GLU (Gated Linear Unit)
        x1, x2 = tf.split(x, 2, axis=-1)
        x = x1 * tf.nn.sigmoid(x2)
        
        x = self.depthwise(x)
        x = self.bn(x, training=training)
        x = tf.nn.silu(x)
        x = self.pointwise2(x)
        x = self.dropout(x, training=training)
        
        return residual + x
    
    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self._d_model, 'kernel_size': self._kernel_size, 'dropout_rate': self._dropout_rate})
        return config


@keras.utils.register_keras_serializable(package='SignBridge')
class ConformerBlock(layers.Layer):
    """
    Full Conformer block:
    FF → MHSA → Conv → FF
    
    Replaces LSTM for better generalization with full batch parallelism.
    """
    
    def __init__(self, d_model=128, num_heads=4, conv_kernel_size=31,
                 ff_expansion=4, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self._d_model = d_model
        self._num_heads = num_heads
        self._conv_kernel_size = conv_kernel_size
        self._ff_expansion = ff_expansion
        self._dropout_rate = dropout_rate
        self.ff1 = ConformerFeedForward(d_model, ff_expansion, dropout_rate)
        self.mhsa_ln = layers.LayerNormalization()
        self.mhsa = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        self.mhsa_dropout = layers.Dropout(dropout_rate)
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout_rate)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion, dropout_rate)
        self.final_ln = layers.LayerNormalization()
    
    def call(self, x, training=False):
        # Feed-forward 1
        x = self.ff1(x, training=training)
        
        # Multi-head self-attention
        residual = x
        x_ln = self.mhsa_ln(x)
        attn_out = self.mhsa(x_ln, x_ln, training=training)
        x = residual + self.mhsa_dropout(attn_out, training=training)
        
        # Convolution module
        x = self.conv_module(x, training=training)
        
        # Feed-forward 2
        x = self.ff2(x, training=training)
        
        # Final layer norm
        x = self.final_ln(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self._d_model,
            'num_heads': self._num_heads,
            'conv_kernel_size': self._conv_kernel_size,
            'ff_expansion': self._ff_expansion,
            'dropout_rate': self._dropout_rate,
        })
        return config


# ============================================================
# Branch B: Multi-Head Self-Attention with Positional Encoding
# ============================================================

def build_attention_branch(input_shape, d_model=128, num_heads=4):
    """
    Branch B: Temporal attention (transformer-style).
    
    Input:  (batch, seq_len, features)
    Output: (batch, seq_len, d_model)
    """
    inputs = layers.Input(shape=input_shape)
    
    # Project to d_model
    x = layers.Dense(d_model)(inputs)
    
    # Add positional encoding
    x = PositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)
    
    # Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=0.1
    )(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    return Model(inputs, x, name='attention_branch')


# ============================================================
# Full TMS-Attention Model
# ============================================================

def build_tms_model(sequence_length: int = 64, n_features: int = 126,
                    n_classes: int = 100, d_model: int = 128,
                    num_heads: int = 4, dropout_rate: float = 0.3,
                    label_smoothing: float = 0.1):
    """
    Build the full TMS-Attention model with Conformer block.
    
    Args:
        sequence_length: Number of frames in input sequence (default 64)
        n_features: Number of features per frame (126 = 2 hands × 21 × 3)
        n_classes: Number of sign classes
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout_rate: Dropout rate for classification head
        label_smoothing: Label smoothing factor
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(sequence_length, n_features), name='landmarks')
    
    # ---- Branch A: Depthwise Separable Convolutions ----
    conv_branch = build_conv_branch((sequence_length, n_features), d_model)
    conv_out = conv_branch(inputs)  # (batch, 64, 128)
    
    # ---- Conformer Block (between Conv and Attention) ----
    conformer = ConformerBlock(
        d_model=d_model, num_heads=num_heads,
        conv_kernel_size=15,  # smaller kernel for short sequences
        dropout_rate=0.1
    )
    conv_out = conformer(conv_out)  # (batch, 64, 128)
    
    # ---- Branch B: Multi-Head Self-Attention ----
    attn_branch = build_attention_branch((sequence_length, n_features), d_model, num_heads)
    attn_out = attn_branch(inputs)  # (batch, 64, 128)
    
    # ---- Concatenate branches ----
    merged = layers.Concatenate()([conv_out, attn_out])  # (batch, 64, 256)
    
    # ---- Classification Head ----
    x = layers.GlobalAveragePooling1D()(merged)  # (batch, 256)
    x = layers.Dense(d_model, activation='relu')(x)  # (batch, 128)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='classification')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TMS_Attention')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
    )
    
    return model


# ============================================================
# CTC Model Wrapper (for continuous signing)
# ============================================================

def build_ctc_model(sequence_length: int = 64, n_features: int = 126,
                    n_classes: int = 100, d_model: int = 128):
    """
    Build CTC-wrapped model for continuous sign recognition.
    
    The base model outputs per-frame class probabilities,
    and CTC loss handles alignment between input frames and label sequence.
    """
    inputs = layers.Input(shape=(sequence_length, n_features), name='landmarks')
    
    # Shared feature extraction
    conv_branch = build_conv_branch((sequence_length, n_features), d_model)
    conv_out = conv_branch(inputs)
    
    conformer = ConformerBlock(d_model=d_model, num_heads=4, conv_kernel_size=15)
    conv_out = conformer(conv_out)
    
    attn_branch = build_attention_branch((sequence_length, n_features), d_model, 4)
    attn_out = attn_branch(inputs)
    
    merged = layers.Concatenate()([conv_out, attn_out])  # (batch, 64, 256)
    
    # Per-frame classification (for CTC)
    x = layers.Dense(d_model, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    # +1 for CTC blank label
    outputs = layers.Dense(n_classes + 1, activation='softmax', name='ctc_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TMS_Attention_CTC')
    return model


# ============================================================
# Model Summary (for testing)
# ============================================================

if __name__ == '__main__':
    print("Building TMS-Attention model (100 classes)...")
    model = build_tms_model(n_classes=100)
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Test forward pass
    dummy_input = np.random.randn(2, 64, 126).astype(np.float32)
    output = model.predict(dummy_input, verbose=0)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sums:  {output.sum(axis=1)}")  # Should be ~1.0
    
    print("\nModel architecture built successfully!")
