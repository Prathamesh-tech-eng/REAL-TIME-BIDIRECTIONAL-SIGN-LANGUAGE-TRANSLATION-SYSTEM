"""
preprocess.py
SignBridge ML Pipeline — Video → Landmark Preprocessing

For each video clip:
  1. Extract frames at 30 FPS using OpenCV
  2. Run MediaPipe Hand Landmarker (Tasks API) on each frame
  3. Extract 21×3 = 63 float values per hand (pad to 2 hands = 126)
  4. Apply landmark normalization (Python port of C++ normalizer)
  5. Apply Kalman filter smoothing
  6. Save as .npy arrays: shape (L, 126) where L = fixed sequence length

Usage:
  python preprocess.py --input_dir data/raw/WLASL100 --output_dir data/processed
                       --sequence_length 64 --fps 30
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


# ============================================================
# Python ports of C++ core functions (for training pipeline)
# ============================================================

def normalize_landmarks_py(raw: np.ndarray) -> np.ndarray:
    """
    Normalize 21 hand landmarks (Python port of C++ normalize_landmarks).
    
    Args:
        raw: np.ndarray shape (21, 3) — raw landmarks from MediaPipe
    Returns:
        np.ndarray shape (21, 3) — normalized landmarks
    """
    out = raw.copy()
    
    # 1. Translate: subtract wrist position
    wrist = out[0].copy()
    out -= wrist
    
    # 2. Scale: divide by wrist-to-MCP9 distance
    scale = np.linalg.norm(out[9])
    if scale < 1e-6:
        scale = 1.0
    out /= scale
    
    return out


class KalmanFilter1D:
    """
    1D Kalman filter (Python port of C++ KalmanFilter1D).
    Default: Q=0.001, R=0.01, P=1.0
    """
    def __init__(self, Q=0.001, R=0.01, P=1.0):
        self.Q = Q
        self.R = R
        self.P = P
        self.x = 0.0
    
    def update(self, z: float) -> float:
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1.0 - K) * self.P
        return self.x
    
    def reset(self):
        self.P = 1.0
        self.x = 0.0


class LandmarkKalmanBank:
    """Bank of 63 Kalman filters for one hand (21 landmarks × 3 axes)."""
    def __init__(self):
        self.filters = [[KalmanFilter1D() for _ in range(3)] for _ in range(21)]
    
    def update(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Args:
            landmarks: shape (21, 3)
        Returns:
            smoothed: shape (21, 3)
        """
        smoothed = np.zeros_like(landmarks)
        for i in range(21):
            for j in range(3):
                smoothed[i, j] = self.filters[i][j].update(float(landmarks[i, j]))
        return smoothed
    
    def reset(self):
        for i in range(21):
            for j in range(3):
                self.filters[i][j].reset()


# ============================================================
# MediaPipe Hand Landmark Extraction
# ============================================================

def extract_landmarks_from_video(video_path: str, fps: int = 30) -> list:
    """
    Extract hand landmarks from a video file using MediaPipe.
    
    Args:
        video_path: Path to video file
        fps: Target FPS for extraction
    
    Returns:
        List of frames, each a dict with 'left' and 'right' hand landmarks
        (each np.ndarray shape (21,3) or None if hand not detected)
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        print("WARNING: MediaPipe not available. Using dummy extraction.")
        return _extract_landmarks_dummy(video_path, fps)
    
    # Initialize Hand Landmarker
    model_path = _get_mediapipe_model_path()
    if model_path is None:
        print("WARNING: MediaPipe model not found. Using dummy extraction.")
        return _extract_landmarks_dummy(video_path, fps)
    
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    
    landmarker = vision.HandLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return []
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0
    
    frame_interval = max(1, int(source_fps / fps))
    frames_data = []
    frame_idx = 0
    timestamp_ms = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += int(1000 / fps)
            
            left_hand = None
            right_hand = None
            
            if result.hand_landmarks:
                for i, handedness in enumerate(result.handedness):
                    hand_label = handedness[0].category_name.lower()
                    landmarks = np.array([
                        [lm.x, lm.y, lm.z] 
                        for lm in result.hand_landmarks[i]
                    ], dtype=np.float32)
                    
                    if hand_label == 'left':
                        left_hand = landmarks
                    else:
                        right_hand = landmarks
            
            frames_data.append({
                'left': left_hand,
                'right': right_hand
            })
        
        frame_idx += 1
    
    cap.release()
    landmarker.close()
    return frames_data


def _get_mediapipe_model_path() -> str:
    """Find the MediaPipe hand landmarker model file."""
    candidates = [
        'hand_landmarker.task',
        'models/hand_landmarker.task',
        os.path.join(os.path.dirname(__file__), '..', 'models', 'hand_landmarker.task'),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _extract_landmarks_dummy(video_path: str, fps: int) -> list:
    """Dummy extraction for when MediaPipe model is not available."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / source_fps
    n_frames = int(duration * fps)
    cap.release()
    
    frames_data = []
    for _ in range(max(n_frames, 1)):
        frames_data.append({
            'left': np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5,
            'right': np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5,
        })
    return frames_data


# ============================================================
# Full Preprocessing Pipeline
# ============================================================

def process_single_video(video_path: str, sequence_length: int = 64,
                         fps: int = 30) -> np.ndarray:
    """
    Full preprocessing pipeline for a single video.
    
    Returns:
        np.ndarray shape (sequence_length, 126) or None if extraction fails
    """
    # Extract landmarks
    frames_data = extract_landmarks_from_video(video_path, fps)
    if not frames_data:
        return None
    
    # Initialize Kalman banks for each hand
    kalman_left = LandmarkKalmanBank()
    kalman_right = LandmarkKalmanBank()
    
    processed_frames = []
    
    for frame in frames_data:
        # Process left hand
        if frame['left'] is not None:
            left_norm = normalize_landmarks_py(frame['left'])
            left_smooth = kalman_left.update(left_norm)
        else:
            left_smooth = np.zeros((21, 3), dtype=np.float32)
        
        # Process right hand
        if frame['right'] is not None:
            right_norm = normalize_landmarks_py(frame['right'])
            right_smooth = kalman_right.update(right_norm)
        else:
            right_smooth = np.zeros((21, 3), dtype=np.float32)
        
        # Concatenate both hands: (21*3 + 21*3) = 126
        frame_vector = np.concatenate([
            left_smooth.flatten(),
            right_smooth.flatten()
        ])
        processed_frames.append(frame_vector)
    
    # Stack to (T, 126)
    sequence = np.array(processed_frames, dtype=np.float32)
    
    # Pad or truncate to fixed sequence length
    if len(sequence) < sequence_length:
        padding = np.zeros((sequence_length - len(sequence), 126), dtype=np.float32)
        sequence = np.concatenate([sequence, padding], axis=0)
    elif len(sequence) > sequence_length:
        sequence = sequence[:sequence_length]
    
    return sequence


def preprocess_dataset(input_dir: str, output_dir: str, 
                       sequence_length: int = 64, fps: int = 30):
    """
    Preprocess an entire dataset directory.
    
    Expected structure:
      input_dir/
        class_0/
          video1.mp4
          video2.mp4
        class_1/
          ...
    
    Output:
      output_dir/
        sequences.npy     — shape (N, sequence_length, 126)
        labels.npy        — shape (N,) integer labels
        label_map.json    — {class_name: int_label}
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Discover classes
    class_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    if not class_dirs:
        print(f"ERROR: No class directories found in {input_dir}")
        return
    
    label_map = {d.name: i for i, d in enumerate(class_dirs)}
    
    all_sequences = []
    all_labels = []
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        label = label_map[class_dir.name]
        videos = [f for f in class_dir.iterdir() 
                  if f.suffix.lower() in video_extensions]
        
        for video_file in tqdm(videos, desc=f"  {class_dir.name}", leave=False):
            sequence = process_single_video(
                str(video_file), sequence_length, fps
            )
            if sequence is not None:
                all_sequences.append(sequence)
                all_labels.append(label)
    
    if not all_sequences:
        print("ERROR: No sequences processed. Check input directory.")
        return
    
    # Save
    sequences_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    np.save(output_path / 'sequences.npy', sequences_array)
    np.save(output_path / 'labels.npy', labels_array)
    
    with open(output_path / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    print(f"\nPreprocessing complete:")
    print(f"  Sequences: {sequences_array.shape}")
    print(f"  Labels:    {labels_array.shape}")
    print(f"  Classes:   {len(label_map)}")
    print(f"  Saved to:  {output_path}")


# ============================================================
# Synthetic Dataset Generator (for pipeline validation)
# ============================================================

def generate_synthetic_dataset(output_dir: str, n_classes: int = 100,
                               samples_per_class: int = 20,
                               sequence_length: int = 64):
    """
    Generate a synthetic landmark dataset for pipeline validation.
    Each class gets a unique "signature" pattern in the landmark space.
    
    This is used to validate the full training pipeline before
    real data (WLASL100, INCLUDE) is available.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate class prototypes
    class_prototypes = []
    for c in range(n_classes):
        # Each class has a distinct temporal pattern
        prototype = np.random.randn(sequence_length, 126).astype(np.float32) * 0.5
        # Add class-specific temporal dynamics
        freq = 0.1 + (c / n_classes) * 0.5
        phase = c * 2 * np.pi / n_classes
        for t in range(sequence_length):
            modulation = np.sin(freq * t + phase) * 0.3
            prototype[t, :63] += modulation  # left hand
            prototype[t, 63:] -= modulation  # right hand (inverted)
        class_prototypes.append(prototype)
    
    all_sequences = []
    all_labels = []
    
    for c in range(n_classes):
        for s in range(samples_per_class):
            # Add noise to prototype
            noise = np.random.randn(sequence_length, 126).astype(np.float32) * 0.1
            sequence = class_prototypes[c] + noise
            all_sequences.append(sequence)
            all_labels.append(c)
    
    sequences_array = np.array(all_sequences, dtype=np.float32)
    labels_array = np.array(all_labels, dtype=np.int32)
    
    # Shuffle
    idx = np.random.permutation(len(all_sequences))
    sequences_array = sequences_array[idx]
    labels_array = labels_array[idx]
    
    # Split: 80/10/10
    n = len(sequences_array)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    splits = {
        'train': (sequences_array[:n_train], labels_array[:n_train]),
        'val': (sequences_array[n_train:n_train+n_val], labels_array[n_train:n_train+n_val]),
        'test': (sequences_array[n_train+n_val:], labels_array[n_train+n_val:]),
    }
    
    label_map = {f"sign_{i:03d}": i for i in range(n_classes)}
    
    for split_name, (seqs, labs) in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)
        np.save(split_dir / 'sequences.npy', seqs)
        np.save(split_dir / 'labels.npy', labs)
    
    with open(output_path / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    # Also save a small representative dataset for quantization
    rep_indices = np.random.choice(n_train, min(200, n_train), replace=False)
    rep_dataset = sequences_array[rep_indices]
    np.save(output_path / 'rep_dataset.npy', rep_dataset)
    
    print(f"Synthetic dataset generated:")
    print(f"  Classes:     {n_classes}")
    print(f"  Train:       {splits['train'][0].shape}")
    print(f"  Validation:  {splits['val'][0].shape}")
    print(f"  Test:        {splits['test'][0].shape}")
    print(f"  Rep dataset: {rep_dataset.shape}")
    print(f"  Saved to:    {output_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignBridge Landmark Preprocessing')
    subparsers = parser.add_subparsers(dest='command')
    
    # Real dataset preprocessing
    proc = subparsers.add_parser('process', help='Process video dataset')
    proc.add_argument('--input_dir', required=True, help='Input video directory')
    proc.add_argument('--output_dir', required=True, help='Output .npy directory')
    proc.add_argument('--sequence_length', type=int, default=64)
    proc.add_argument('--fps', type=int, default=30)
    
    # Synthetic dataset generation
    syn = subparsers.add_parser('synthetic', help='Generate synthetic dataset')
    syn.add_argument('--output_dir', required=True, help='Output directory')
    syn.add_argument('--n_classes', type=int, default=100)
    syn.add_argument('--samples_per_class', type=int, default=20)
    syn.add_argument('--sequence_length', type=int, default=64)
    
    args = parser.parse_args()
    
    if args.command == 'process':
        preprocess_dataset(args.input_dir, args.output_dir,
                          args.sequence_length, args.fps)
    elif args.command == 'synthetic':
        generate_synthetic_dataset(args.output_dir, args.n_classes,
                                  args.samples_per_class, args.sequence_length)
    else:
        parser.print_help()
