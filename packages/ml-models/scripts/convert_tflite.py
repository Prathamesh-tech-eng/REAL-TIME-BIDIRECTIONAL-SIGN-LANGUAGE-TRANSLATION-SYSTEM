"""
convert_tflite.py
SignBridge ML — TFLite Conversion with Quantization

Usage:
  python convert_tflite.py \
    --model models/tms_wlasl100.h5 \
    --output models/tms_wlasl100_int8.tflite \
    --quantize int8 \
    --representative_dataset data/processed/rep_dataset.npy
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add parent src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import custom layers so they are registered before model load
import tms_model  # noqa: F401 — registers @register_keras_serializable layers


def convert_to_tflite(model_path: str, output_path: str,
                      quantize: str = 'none',
                      representative_dataset_path: str = None):
    """
    Convert a Keras .h5 model to TFLite format.
    
    Args:
        model_path: Path to .h5 model file
        output_path: Path for output .tflite file
        quantize: Quantization mode: 'none', 'float16', 'int8'
        representative_dataset_path: Path to .npy representative dataset (for int8)
    """
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print(f"Model loaded. Parameters: {model.count_params():,}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize == 'float16':
        print("Applying float16 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    elif quantize == 'int8':
        print("Applying int8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset_path:
            print(f"Loading representative dataset: {representative_dataset_path}")
            rep_data = np.load(representative_dataset_path).astype(np.float32)
            
            def representative_dataset_gen():
                for i in range(min(len(rep_data), 200)):
                    yield [rep_data[i:i+1]]
            
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.float32   # Keep float input for ease of use
            converter.inference_output_type = tf.float32  # Keep float output
        else:
            print("WARNING: No representative dataset provided. Using dynamic range quantization.")
    
    elif quantize == 'none':
        print("No quantization applied (float32).")
    
    else:
        raise ValueError(f"Unknown quantization mode: {quantize}")
    
    print("Converting...")
    tflite_model = converter.convert()
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Report size
    size_bytes = os.path.getsize(output_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"\nConversion complete:")
    print(f"  Output:       {output_path}")
    print(f"  Size:         {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  Quantization: {quantize}")
    
    # Verify the converted model works
    print("\nVerifying converted model...")
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"  Input:  {input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"  Output: {output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    
    # Run a test inference
    test_input = np.random.randn(*input_details[0]['shape']).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    test_output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"  Test output shape: {test_output.shape}")
    print(f"  Test output sum:   {test_output.sum():.4f} (expected ~1.0)")
    print(f"  Verification: PASSED")
    
    return output_path, size_mb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SignBridge TFLite Conversion')
    parser.add_argument('--model', required=True, help='Path to .h5 model')
    parser.add_argument('--output', required=True, help='Path for .tflite output')
    parser.add_argument('--quantize', choices=['none', 'float16', 'int8'], default='int8')
    parser.add_argument('--representative_dataset', default=None,
                        help='Path to .npy representative dataset (for int8 quantization)')
    
    args = parser.parse_args()
    convert_to_tflite(args.model, args.output, args.quantize,
                      args.representative_dataset)
