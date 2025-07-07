#!/usr/bin/env python3
"""
Comprehensive example of eddy detection and tracking using the trained DeepLabv3+ model

This script demonstrates:
1. Loading the trained model architecture and weights
2. Preprocessing SSH images to match training format
3. Running inference to get segmentation masks
4. Extracting eddy contours and centroids
5. Tracking eddies across frames
6. Visualizing results with contours and labels
7. Saving annotated output images

Based on the FrameWork.py training script and the user's requirements.
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Activation, Concatenate, Add, Dropout
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, DepthwiseConv2D
from tensorflow.keras.layers import ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import get_source_inputs
from tensorflow.keras import backend as K
from scipy.spatial.distance import cdist
import tensorflow as tf

# Import the model architecture from FrameWork.py
from FrameWork import Deeplabv3, BilinearUpsampling, SepConv_BN, _conv2d_same, _xception_block, relu6, _make_divisible, _inverted_res_block

def load_trained_model(weights_path='./eddydlv3net/eddynet.weights.h5', input_shape=(80, 84, 1)):
    """
    Load the trained DeepLabv3+ model with the same architecture as training
    
    Args:
        weights_path: Path to the trained model weights
        input_shape: Input shape for the model (height, width, channels)
    
    Returns:
        Loaded model ready for inference
    """
    print("Building model architecture (same as training)...")
    
    # Build the model architecture (exactly as in FrameWork.py)
    model = Deeplabv3(input_shape=input_shape, classes=3)
    
    print(f"Loading trained weights from {weights_path}...")
    try:
        model.load_weights(weights_path)
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        raise
    
    return model

def preprocess_ssh_image(image, target_shape=(80, 84)):
    """
    Preprocess SSH image to match training format
    
    Args:
        image: Input SSH image (numpy array)
        target_shape: Target shape (height, width) for the model
    
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure image is float32
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    
    # Resize to model input size if needed
    if image.shape[:2] != target_shape:
        print(f"Resizing image from {image.shape[:2]} to {target_shape}")
        image = cv2.resize(image, (target_shape[1], target_shape[0]))
    
    # Add channel dimension if needed
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def run_inference(model, image):
    """
    Run inference on SSH image
    
    Args:
        model: Loaded DeepLabv3+ model
        image: Input SSH image (numpy array)
    
    Returns:
        prediction: Raw model output (probabilities)
        mask: Segmentation mask (class indices)
    """
    # Preprocess image
    input_tensor = preprocess_ssh_image(image)
    
    # Run prediction
    prediction = model.predict(input_tensor, verbose=0)
    
    # Convert to segmentation mask
    mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
    
    return prediction[0], mask

def extract_eddy_contours_and_centroids(mask):
    """
    Extract eddy contours and centroids from segmentation mask
    
    Args:
        mask: Segmentation mask with class indices (0=background, 1=cyclonic, 2=anticyclonic)
    
    Returns:
        List of eddy dictionaries with centroids, classes, and contours
    """
    eddies = []
    
    # Process each eddy class (1=cyclonic, 2=anticyclonic)
    for class_id in [1, 2]:
        # Create binary mask for this class
        binary = (mask == class_id).astype(np.uint8) * 255
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate area and filter small contours
            area = cv2.contourArea(contour)
            if area < 10:  # Minimum area threshold
                continue
            
            # Calculate centroid using moments
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                eddies.append({
                    'centroid': (cx, cy),
                    'class': class_id,
                    'contour': contour,
                    'area': area
                })
    
    return eddies

def track_eddies_across_frames(prev_eddies, curr_eddies, max_distance=20):
    """
    Track eddies across frames using centroid matching
    
    Args:
        prev_eddies: List of eddies from previous frame with IDs
        curr_eddies: List of eddies from current frame
        max_distance: Maximum distance for tracking (pixels)
    
    Returns:
        Updated list of tracked eddies with IDs
    """
    if not prev_eddies:
        # First frame: assign new IDs to all eddies
        for i, eddy in enumerate(curr_eddies):
            eddy['id'] = i + 1
        return curr_eddies
    
    if not curr_eddies:
        return []
    
    # Calculate distance matrix between previous and current eddies
    prev_centroids = [e['centroid'] for e in prev_eddies]
    curr_centroids = [e['centroid'] for e in curr_eddies]
    
    distances = cdist(prev_centroids, curr_centroids)
    
    # Match eddies based on minimum distance
    matched_prev = set()
    matched_curr = set()
    tracked_eddies = []
    
    # Find best matches
    while True:
        if len(matched_prev) == len(prev_centroids) or len(matched_curr) == len(curr_eddies):
            break
        
        # Find minimum distance among unmatched pairs
        min_dist = float('inf')
        min_i, min_j = -1, -1
        
        for i in range(len(prev_centroids)):
            if i in matched_prev:
                continue
            for j in range(len(curr_centroids)):
                if j in matched_curr:
                    continue
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    min_i, min_j = i, j
        
        # If minimum distance is too large, stop matching
        if min_dist > max_distance:
            break
        
        # Match this pair
        matched_prev.add(min_i)
        matched_curr.add(min_j)
        
        # Update tracked eddy with new information but keep ID
        tracked_eddy = prev_eddies[min_i].copy()
        tracked_eddy.update(curr_eddies[min_j])
        tracked_eddies.append(tracked_eddy)
    
    # Add new eddies (unmatched current eddies)
    next_id = max([e['id'] for e in prev_eddies]) + 1 if prev_eddies else 1
    for j, eddy in enumerate(curr_eddies):
        if j not in matched_curr:
            eddy['id'] = next_id
            next_id += 1
            tracked_eddies.append(eddy)
    
    return tracked_eddies

def visualize_eddy_detection(image, mask, tracked_eddies=None, save_path=None):
    """
    Visualize eddy detection results with contours and labels
    
    Args:
        image: Original SSH image
        mask: Segmentation mask
        tracked_eddies: List of tracked eddies with IDs
        save_path: Path to save visualization
    
    Returns:
        Annotated image
    """
    # Convert to BGR for OpenCV
    if len(image.shape) == 2:
        # Convert grayscale to BGR
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()
    
    # Normalize image for visualization
    if annotated.dtype == np.float32:
        annotated = ((annotated - annotated.min()) / (annotated.max() - annotated.min()) * 255).astype(np.uint8)
    
    # Define colors for each class
    colors = {
        1: (0, 0, 255),    # Cyclonic - red
        2: (0, 255, 0)     # Anticyclonic - green
    }
    
    # Draw contours for each class
    for class_id in [1, 2]:  # Cyclonic and Anticyclonic
        binary = (mask == class_id).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = colors[class_id]
        cv2.drawContours(annotated, contours, -1, color, 2)
    
    # Add labels and IDs if tracking data is available
    if tracked_eddies:
        for eddy in tracked_eddies:
            cx, cy = eddy['centroid']
            class_id = eddy['class']
            eddy_id = eddy.get('id', '?')
            
            # Create label
            class_name = "AC" if class_id == 2 else "CE"  # Anticyclonic/Cyclonic
            label = f"{class_name}{eddy_id}"
            
            # Choose text color (white for visibility)
            text_color = (255, 255, 255)
            
            # Add text background for better visibility
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(annotated, 
                        (cx - text_size[0]//2 - 2, cy - text_size[1] - 2),
                        (cx + text_size[0]//2 + 2, cy + 2),
                        (0, 0, 0), -1)
            
            # Add text
            cv2.putText(annotated, label, 
                       (cx - text_size[0]//2, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    # Save if path provided
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved visualization to: {save_path}")
    
    return annotated

def process_ssh_sequence(model, image_paths, output_dir='./output'):
    """
    Process a sequence of SSH images for eddy detection and tracking
    
    Args:
        model: Loaded DeepLabv3+ model
        image_paths: List of paths to SSH images
        output_dir: Directory to save output visualizations
    
    Returns:
        List of tracked eddies from the last frame
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(image_paths)} images...")
    
    tracked_eddies = []  # Track eddies across frames
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path}")
        
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
        
        # Run inference
        prediction, mask = run_inference(model, image)
        
        # Extract eddies
        current_eddies = extract_eddy_contours_and_centroids(mask)
        
        # Track eddies
        tracked_eddies = track_eddies_across_frames(tracked_eddies, current_eddies)
        
        # Visualize results
        output_path = os.path.join(output_dir, f'frame_{i:03d}.png')
        annotated = visualize_eddy_detection(image, mask, tracked_eddies, output_path)
        
        # Print detection summary
        num_cyclonic = sum(1 for e in tracked_eddies if e['class'] == 1)
        num_anticyclonic = sum(1 for e in tracked_eddies if e['class'] == 2)
        print(f"  Detected: {num_cyclonic} cyclonic, {num_anticyclonic} anticyclonic eddies")
    
    print(f"✅ Processing complete! Results saved to {output_dir}")
    return tracked_eddies

def main():
    """
    Main function demonstrating the complete eddy detection and tracking pipeline
    """
    print("=== Eddy Detection and Tracking Example ===\n")
    
    # 1. Load the trained model
    print("1. Loading trained model...")
    model = load_trained_model()
    
    # 2. Load test data from the Sea directory
    print("\n2. Loading test data...")
    data_dir = r'C:/Users/jalla/Downloads/Eddy/Sea'
    ssh_test = np.load(os.path.join(data_dir, 'filtered_SSH_test_data.npy'))
    print(f"   Loaded test data with shape: {ssh_test.shape}")
    
    # 3. Process single image example
    print("\n3. Processing single image example...")
    test_image = ssh_test[0, :80, :84]  # Use same dimensions as training
    
    # Run inference
    prediction, mask = run_inference(model, test_image)
    
    # Extract eddies
    eddies = extract_eddy_contours_and_centroids(mask)
    
    # Assign IDs for single image
    for i, eddy in enumerate(eddies):
        eddy['id'] = i + 1
    
    # Visualize single image
    output_dir = './eddydlv3net/example_output'
    os.makedirs(output_dir, exist_ok=True)
    
    single_output_path = os.path.join(output_dir, 'single_frame_detection.png')
    annotated = visualize_eddy_detection(test_image, mask, eddies, single_output_path)
    
    print(f"   Single frame processing complete. Results saved to {single_output_path}")
    print(f"   Detected {len(eddies)} eddies")
    
    # 4. Process image sequence example
    print("\n4. Processing image sequence example...")
    
    # Create a sequence from test data (first 10 frames)
    sequence_images = []
    for i in range(min(10, len(ssh_test))):
        sequence_images.append(ssh_test[i, :80, :84])
    
    # Process sequence
    sequence_output_dir = os.path.join(output_dir, 'sequence')
    final_tracked_eddies = process_ssh_sequence(model, sequence_images, sequence_output_dir)
    
    print(f"   Sequence processing complete. Final tracked eddies: {len(final_tracked_eddies)}")
    
    # 5. Print summary
    print("\n=== Summary ===")
    print(f"✅ Model loaded successfully")
    print(f"✅ Single image processed: {len(eddies)} eddies detected")
    print(f"✅ Image sequence processed: {len(final_tracked_eddies)} final tracked eddies")
    print(f"✅ Results saved to: {output_dir}")
    
    # 6. Show example of eddy information
    if final_tracked_eddies:
        print(f"\nExample tracked eddy information:")
        for eddy in final_tracked_eddies[:3]:  # Show first 3 eddies
            class_name = "Anticyclonic" if eddy['class'] == 2 else "Cyclonic"
            print(f"   Eddy {eddy['id']}: {class_name} at {eddy['centroid']}, area: {eddy['area']:.1f}")

if __name__ == "__main__":
    main() 