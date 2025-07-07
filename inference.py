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

class EddyDetector:
    def __init__(self, weights_path='./eddydlv3net/eddynet.weights.h5', input_shape=(80, 84, 1)):
        """
        Initialize the Eddy Detector with trained model
        
        Args:
            weights_path: Path to the trained model weights
            input_shape: Input shape for the model (height, width, channels)
        """
        self.input_shape = input_shape
        self.height, self.width = input_shape[0], input_shape[1]
        
        # Build the model architecture (same as training)
        print("Building model architecture...")
        self.model = Deeplabv3(input_shape=input_shape, classes=3)
        
        # Load trained weights
        print(f"Loading weights from {weights_path}...")
        try:
            self.model.load_weights(weights_path)
            print("✅ Model weights loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            raise
        
        # Initialize tracking variables
        self.tracked_eddies = []  # List of dictionaries with 'id', 'centroid', 'class', 'frames_missing'
        self.next_id = 1
        self.max_distance = 20  # Maximum distance for tracking (pixels)
        self.max_frames_missing = 5  # Maximum frames an eddy can be missing before removing
        
        # Define colors for visualization
        self.colors = {
            0: (128, 128, 128),  # Background - gray
            1: (0, 0, 255),      # Cyclonic - red
            2: (0, 255, 0)       # Anticyclonic - green
        }
        
        self.class_names = {
            0: "Background",
            1: "Cyclonic",
            2: "Anticyclonic"
        }
    
    def preprocess_image(self, image):
        """
        Preprocess SSH image to match training format
        
        Args:
            image: Input SSH image (numpy array)
            
        Returns:
            Preprocessed image ready for model input
        """
        # Ensure image is float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Resize to model input size if needed
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(image, (self.width, self.height))
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = image[..., np.newaxis]
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """
        Run inference on SSH image
        
        Args:
            image: Input SSH image (numpy array)
            
        Returns:
            prediction: Raw model output (probabilities)
            mask: Segmentation mask (class indices)
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        # Run prediction
        prediction = self.model.predict(input_tensor, verbose=0)
        
        # Convert to segmentation mask
        mask = np.argmax(prediction[0], axis=-1).astype(np.uint8)
        
        return prediction[0], mask
    
    def extract_eddies(self, mask):
        """
        Extract eddy contours and centroids from segmentation mask
        
        Args:
            mask: Segmentation mask with class indices
            
        Returns:
            List of eddy dictionaries with centroids and classes
        """
        eddies = []
        
        # Process each eddy class (1=cyclonic, 2=anticyclonic)
        for class_id in [1, 2]:
            # Create binary mask for this class
            binary = (mask == class_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate area and filter small contours
                area = cv2.contourArea(contour)
                if area < 10:  # Minimum area threshold
                    continue
                
                # Calculate centroid
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
    
    def track_eddies(self, current_eddies):
        """
        Track eddies across frames using centroid matching
        
        Args:
            current_eddies: List of eddies detected in current frame
            
        Returns:
            Updated list of tracked eddies with IDs
        """
        # Update missing frames for existing tracked eddies
        for eddy in self.tracked_eddies:
            eddy['frames_missing'] += 1
        
        # Remove eddies that have been missing for too long
        self.tracked_eddies = [e for e in self.tracked_eddies if e['frames_missing'] <= self.max_frames_missing]
        
        # If no current eddies, return empty list
        if not current_eddies:
            return []
        
        # If no tracked eddies, assign new IDs to all current eddies
        if not self.tracked_eddies:
            for eddy in current_eddies:
                eddy['id'] = self.next_id
                eddy['frames_missing'] = 0
                self.next_id += 1
            self.tracked_eddies = current_eddies
            return current_eddies
        
        # Calculate distance matrix between tracked and current eddies
        tracked_centroids = [e['centroid'] for e in self.tracked_eddies]
        current_centroids = [e['centroid'] for e in current_eddies]
        
        if tracked_centroids and current_centroids:
            distances = cdist(tracked_centroids, current_centroids)
            
            # Match eddies based on minimum distance
            matched_tracked = set()
            matched_current = set()
            
            # Find best matches
            while True:
                if len(matched_tracked) == len(tracked_centroids) or len(matched_current) == len(current_eddies):
                    break
                
                # Find minimum distance among unmatched pairs
                min_dist = float('inf')
                min_i, min_j = -1, -1
                
                for i in range(len(tracked_centroids)):
                    if i in matched_tracked:
                        continue
                    for j in range(len(current_centroids)):
                        if j in matched_current:
                            continue
                        if distances[i, j] < min_dist:
                            min_dist = distances[i, j]
                            min_i, min_j = i, j
                
                # If minimum distance is too large, stop matching
                if min_dist > self.max_distance:
                    break
                
                # Match this pair
                matched_tracked.add(min_i)
                matched_current.add(min_j)
                
                # Update tracked eddy
                self.tracked_eddies[min_i].update(current_eddies[min_j])
                self.tracked_eddies[min_i]['frames_missing'] = 0
            
            # Add new eddies (unmatched current eddies)
            for j, eddy in enumerate(current_eddies):
                if j not in matched_current:
                    eddy['id'] = self.next_id
                    eddy['frames_missing'] = 0
                    self.next_id += 1
                    self.tracked_eddies.append(eddy)
        
        return self.tracked_eddies
    
    def visualize_detection(self, image, mask, tracked_eddies=None, save_path=None):
        """
        Visualize eddy detection results
        
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
        
        # Draw contours for each class
        for class_id in [1, 2]:  # Cyclonic and Anticyclonic
            binary = (mask == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color = self.colors[class_id]
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
    
    def process_sequence(self, image_paths, output_dir='./output'):
        """
        Process a sequence of SSH images for eddy detection and tracking
        
        Args:
            image_paths: List of paths to SSH images
            output_dir: Directory to save output visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {len(image_paths)} images...")
        
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
            
            # Run detection
            prediction, mask = self.predict(image)
            
            # Extract eddies
            current_eddies = self.extract_eddies(mask)
            
            # Track eddies
            tracked_eddies = self.track_eddies(current_eddies)
            
            # Visualize results
            output_path = os.path.join(output_dir, f'frame_{i:03d}.png')
            annotated = self.visualize_detection(image, mask, tracked_eddies, output_path)
            
            # Print detection summary
            num_cyclonic = sum(1 for e in tracked_eddies if e['class'] == 1)
            num_anticyclonic = sum(1 for e in tracked_eddies if e['class'] == 2)
            print(f"  Detected: {num_cyclonic} cyclonic, {num_anticyclonic} anticyclonic eddies")
        
        print(f"✅ Processing complete! Results saved to {output_dir}")
        return tracked_eddies

def main():
    """
    Example usage of the EddyDetector
    """
    # Initialize detector
    detector = EddyDetector()
    
    # Example 1: Process a single image
    print("\n=== Single Image Processing ===")
    
    # Load test data from the Sea directory
    data_dir = r'C:/Users/jalla/Downloads/Eddy/Sea'
    ssh_test = np.load(os.path.join(data_dir, 'filtered_SSH_test_data.npy'))
    
    # Process first test image
    test_image = ssh_test[0, :80, :84]  # Use same dimensions as training
    
    # Run detection
    prediction, mask = detector.predict(test_image)
    
    # Extract and track eddies
    current_eddies = detector.extract_eddies(mask)
    tracked_eddies = detector.track_eddies(current_eddies)
    
    # Visualize
    output_path = './eddydlv3net/single_frame_detection.png'
    annotated = detector.visualize_detection(test_image, mask, tracked_eddies, output_path)
    
    print(f"Single frame processing complete. Results saved to {output_path}")
    
    # Example 2: Process a sequence of images
    print("\n=== Sequence Processing ===")
    
    # Create a sequence from test data (first 10 frames)
    sequence_images = []
    for i in range(min(10, len(ssh_test))):
        sequence_images.append(ssh_test[i, :80, :84])
    
    # Process sequence
    output_dir = './eddydlv3net/sequence_output'
    final_tracked_eddies = detector.process_sequence(sequence_images, output_dir)
    
    print(f"Sequence processing complete. Final tracked eddies: {len(final_tracked_eddies)}")

if __name__ == "__main__":
    main() 