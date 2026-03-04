"""
train.py
SignBridge ML — Model Training Script

Usage:
  python train.py \
    --dataset data/processed \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --early_stopping_patience 10 \
    --output models/tms_wlasl100.h5
"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Add parent src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from tms_model import build_tms_model


def load_dataset(data_dir: str):
    """
    Load preprocessed dataset from .npy files.
    
    Expected structure:
      data_dir/
        train/sequences.npy, labels.npy
        val/sequences.npy, labels.npy
        test/sequences.npy, labels.npy
        label_map.json
    """
    data_path = Path(data_dir)
    
    # Load label map
    with open(data_path / 'label_map.json', 'r') as f:
        label_map = json.load(f)
    n_classes = len(label_map)
    
    splits = {}
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        sequences = np.load(split_dir / 'sequences.npy')
        labels = np.load(split_dir / 'labels.npy')
        
        # One-hot encode labels
        labels_onehot = keras.utils.to_categorical(labels, num_classes=n_classes)
        
        splits[split] = (sequences, labels_onehot)
        print(f"  {split}: sequences={sequences.shape}, labels={labels_onehot.shape}")
    
    return splits, label_map, n_classes


def augment_data(sequences: np.ndarray, labels: np.ndarray,
                 rotation_range: float = 15.0, time_warp: bool = True):
    """
    Data augmentation for landmark sequences.
    
    - Random rotation ±15° around Z-axis (applied to each hand's landmarks)
    - Time warping: randomly stretch/compress temporal dimension
    - Random noise injection
    """
    augmented_seqs = []
    augmented_labels = []
    
    for seq, label in zip(sequences, labels):
        # Original
        augmented_seqs.append(seq)
        augmented_labels.append(label)
        
        # Augmented copy 1: rotation
        aug1 = seq.copy()
        angle = np.random.uniform(-rotation_range, rotation_range) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for t in range(len(aug1)):
            # Rotate left hand (indices 0-62, every 3 values = x,y,z)
            for i in range(0, 63, 3):
                x, y = aug1[t, i], aug1[t, i+1]
                aug1[t, i] = x * cos_a - y * sin_a
                aug1[t, i+1] = x * sin_a + y * cos_a
            # Rotate right hand (indices 63-125)
            for i in range(63, 126, 3):
                x, y = aug1[t, i], aug1[t, i+1]
                aug1[t, i] = x * cos_a - y * sin_a
                aug1[t, i+1] = x * sin_a + y * cos_a
        
        augmented_seqs.append(aug1)
        augmented_labels.append(label)
        
        # Augmented copy 2: noise
        aug2 = seq + np.random.randn(*seq.shape).astype(np.float32) * 0.02
        augmented_seqs.append(aug2)
        augmented_labels.append(label)
    
    return np.array(augmented_seqs), np.array(augmented_labels)


def train(args):
    """Main training loop."""
    print("="*60)
    print("  SignBridge TMS-Attention Model Training")
    print("="*60)
    
    # Load data
    print(f"\nLoading dataset from: {args.dataset}")
    splits, label_map, n_classes = load_dataset(args.dataset)
    
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    
    print(f"\nDataset summary:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features:        {n_features}")
    print(f"  Classes:         {n_classes}")
    
    # Augment training data
    if args.augment:
        print("\nApplying data augmentation...")
        X_train, y_train = augment_data(X_train, y_train)
        print(f"  After augmentation: {X_train.shape}")
    
    # Build model
    print("\nBuilding TMS-Attention model...")
    model = build_tms_model(
        sequence_length=seq_len,
        n_features=n_features,
        n_classes=n_classes,
        d_model=128,
        num_heads=4,
        dropout_rate=0.3,
        label_smoothing=args.label_smoothing
    )
    
    # Update learning rate if specified
    if args.learning_rate != 0.001:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy')]
        )
    
    try:
        model.summary(line_length=120)
    except ValueError:
        print("  (model.summary skipped — console too narrow)")
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Callbacks
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(output_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]
    
    # Train
    print(f"\nTraining for up to {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Early stopping patience: {args.early_stopping_patience}")
    print(f"  Label smoothing: {args.label_smoothing}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_top5 = test_results
    
    print(f"\n{'='*60}")
    print(f"  Training Complete")
    print(f"{'='*60}")
    print(f"  Best val accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"  Test accuracy:     {test_acc:.4f}")
    print(f"  Test top-5 acc:    {test_top5:.4f}")
    print(f"  Test loss:         {test_loss:.4f}")
    print(f"  Model saved to:    {output_path}")
    
    # Save training history
    history_path = output_path.parent / f"{output_path.stem}_history.json"
    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"  History saved to:  {history_path}")
    
    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignBridge TMS Model Training')
    parser.add_argument('--dataset', required=True, help='Path to preprocessed dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--output', default='models/tms_wlasl100.h5')
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no-augment', dest='augment', action='store_false')
    
    args = parser.parse_args()
    train(args)
