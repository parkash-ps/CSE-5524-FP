#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 Segmentation Model Training and Evaluation

This script performs the following tasks:
1. Installs required packages
2. Trains a YOLOv8 segmentation model on a pre-formatted dataset for 100 epochs
3. Evaluates the model performance
4. Plots training metrics, confusion matrix, and other performance visualizations

Assumes dataset is already formatted in YOLOv8-compatible structure with train/val/test splits
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import yaml
import pandas as pd
from datetime import datetime
from PIL import Image
import cv2
import shutil

# Check if ultralytics is installed, if not, install it
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing required packages...")
    os.system('pip install ultralytics matplotlib seaborn pandas opencv-python')
    from ultralytics import YOLO

# Configure paths
DATASET_PATH = '/Users/parkash/Library/CloudStorage/OneDrive-TheOhioStateUniversity/CV_Course/Mice_project/Mice_dataset'  # Update this to your dataset path
OUTPUT_DIR = 'runs/training_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def check_dataset_structure():
    """Verify that the dataset is properly formatted for YOLOv8."""
    required_dirs = ['train', 'val', 'test']
    required_subdirs = ['images', 'labels']
    
    for dir_name in required_dirs:
        dir_path = os.path.join(DATASET_PATH, dir_name)
        if not os.path.exists(dir_path):
            print(f"Warning: {dir_path} directory not found")
            return False
        
        for subdir in required_subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            if not os.path.exists(subdir_path):
                print(f"Warning: {subdir_path} directory not found")
                return False
    
    # Verify data.yaml exists
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found")
        return False
        
    print("Dataset structure verified successfully!")
    return True


def create_data_yaml():
    """Create data.yaml file if it doesn't exist."""
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
    
    if os.path.exists(yaml_path):
        print(f"Using existing data.yaml at {yaml_path}")
        return yaml_path
    
    # Get class names from directory structure if possible
    # This is a simplistic approach; adjust as needed
    try:
        label_files = list(Path(os.path.join(DATASET_PATH, 'train/labels')).glob('*.txt'))
        class_ids = set()
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
        
        num_classes = max(class_ids) + 1
        class_names = [f"class_{i}" for i in range(num_classes)]
        
    except Exception as e:
        print(f"Error determining class names: {e}")
        print("Using default class names...")
        class_names = ["object"]
    
    # Create the data.yaml file
    data_dict = {
        'path': DATASET_PATH,
        'train': os.path.join(DATASET_PATH, 'train/images'),
        'val': os.path.join(DATASET_PATH, 'val/images'),
        'test': os.path.join(DATASET_PATH, 'test/images'),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def train_model(data_yaml_path):
    """Train YOLOv8 segmentation model for 100 epochs."""
    print("Starting model training...")
    
    # Initialize model - YOLOv8n-seg is a smaller model for faster training
    # You can use 's', 'm', 'l', or 'x' for progressively larger models
    model = YOLO('yolov8n-seg.pt')  
    
    # Set training arguments
    args = {
        'data': data_yaml_path,
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'device': 0 if torch.cuda.is_available() else 'cpu',
        'name': 'segmentation_model',
        'project': OUTPUT_DIR,
        'exist_ok': True,
        'patience': 50,         # Early stopping after 50 epochs with no improvement
        'save': True,           # Save model
        'verbose': True,        # Detailed training information
        'seed': 42,             # For reproducibility
        'cos_lr': True,         # Use cosine learning rate scheduler
        'plots': True,          # Generate plots
    }
    
    # Start training
    results = model.train(**args)
    
    # Return the path to the best model weights
    model_path = str(Path(results.save_dir) / 'weights' / 'best.pt')
    return model_path, results


def evaluate_model(model_path):
    """Evaluate the trained model on validation and test sets."""
    print("Evaluating model performance...")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Evaluate on validation set
    val_results = model.val(
        data=os.path.join(DATASET_PATH, 'val/images'),
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    # Evaluate on test set
    test_results = model.val(
        data=os.path.join(DATASET_PATH, 'test/images'),
        save_json=True,
        save_hybrid=True,
        plots=True
    )
    
    return val_results, test_results


def generate_confusion_matrix(model_path):
    """Generate and plot confusion matrix for the model."""
    print("Generating confusion matrix...")
    
    try:
        import supervision as sv
        
        # Load the model
        model = YOLO(model_path)
        
        # Get validation dataset path
        val_path = os.path.join(DATASET_PATH, 'val')
        
        # Create a DetectionDataset from YOLO format
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=os.path.join(val_path, 'images'),
            annotations_directory_path=os.path.join(val_path, 'labels'),
            data_yaml_path=os.path.join(DATASET_PATH, 'data.yaml')
        )
        
        # Callback function for the model
        def callback(image):
            result = model(image)[0]
            return sv.Detections.from_ultralytics(result)
        
        # Generate confusion matrix
        confusion_matrix = sv.ConfusionMatrix.benchmark(
            dataset=dataset,
            callback=callback
        )
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        confusion_matrix.plot()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
        plt.close()
        
        return True
    
    except ImportError:
        print("Supervision package not installed. Installing now...")
        os.system('pip install supervision')
        print("Please run the script again to generate the confusion matrix.")
        return False
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        print("Generating confusion matrix using alternative method...")
        
        # Alternative confusion matrix generation using ultralytics built-in methods
        # Load the model
        model = YOLO(model_path)
        
        # Validate on test set with confusion matrix
        model.val(
            data=os.path.join(DATASET_PATH, 'val'),
            conf=0.25,
            iou=0.7,
            max_det=300,
            plots=True,
        )
        
        # Copy the confusion matrix if it exists
        confusion_matrix_path = Path('runs') / 'segment' / 'val' / 'confusion_matrix.png'
        if confusion_matrix_path.exists():
            shutil.copy(str(confusion_matrix_path), os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
            return True
        return False


def plot_metrics(results):
    """Plot training metrics from the results object."""
    print("Plotting training metrics...")
    
    # Plot training results
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    
    # Extract metrics from CSV if available
    try:
        results_csv = Path(results.save_dir) / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            
            # Plot loss curves
            ax[0, 0].plot(df['epoch'], df['train/box_loss'], label='train box loss')
            ax[0, 0].plot(df['epoch'], df['val/box_loss'], label='val box loss')
            ax[0, 0].plot(df['epoch'], df['train/seg_loss'], label='train seg loss')
            ax[0, 0].plot(df['epoch'], df['val/seg_loss'], label='val seg loss')
            ax[0, 0].set_xlabel('Epoch')
            ax[0, 0].set_ylabel('Loss')
            ax[0, 0].set_title('Training and Validation Losses')
            ax[0, 0].legend()
            
            # Plot mAP curves
            ax[0, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (Box)')
            ax[0, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (Box)')
            ax[0, 1].plot(df['epoch'], df['metrics/mAP50(M)'], label='mAP50 (Mask)')
            ax[0, 1].plot(df['epoch'], df['metrics/mAP50-95(M)'], label='mAP50-95 (Mask)')
            ax[0, 1].set_xlabel('Epoch')
            ax[0, 1].set_ylabel('mAP')
            ax[0, 1].set_title('Mean Average Precision')
            ax[0, 1].legend()
            
            # Plot precision and recall
            ax[0, 2].plot(df['epoch'], df['metrics/precision(B)'], label='Precision (Box)')
            ax[0, 2].plot(df['epoch'], df['metrics/recall(B)'], label='Recall (Box)')
            ax[0, 2].plot(df['epoch'], df['metrics/precision(M)'], label='Precision (Mask)')
            ax[0, 2].plot(df['epoch'], df['metrics/recall(M)'], label='Recall (Mask)')
            ax[0, 2].set_xlabel('Epoch')
            ax[0, 2].set_ylabel('Value')
            ax[0, 2].set_title('Precision and Recall')
            ax[0, 2].legend()
            
            # Plot learning rate
            if 'lr/pg0' in df.columns:
                ax[1, 0].plot(df['epoch'], df['lr/pg0'], label='Learning Rate')
                ax[1, 0].set_xlabel('Epoch')
                ax[1, 0].set_ylabel('Learning Rate')
                ax[1, 0].set_title('Learning Rate Schedule')
                ax[1, 0].legend()
        
        # Save the plotted results
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_metrics.png'))
        plt.close()
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")
    
    # Copy any plots generated during training
    try:
        plots_dir = Path(results.save_dir) / 'plots'
        if plots_dir.exists():
            for plot_file in plots_dir.glob('*.png'):
                shutil.copy(str(plot_file), os.path.join(OUTPUT_DIR, plot_file.name))
    except Exception as e:
        print(f"Error copying plots: {e}")


def visualize_predictions(model_path, num_samples=5):
    """Visualize model predictions on sample images."""
    print("Visualizing model predictions...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Get test images
    test_images_dir = os.path.join(DATASET_PATH, 'test/images')
    image_files = list(Path(test_images_dir).glob('*.jpg')) + list(Path(test_images_dir).glob('*.png'))
    
    if not image_files:
        print("No test images found!")
        return
    
    # Select random samples
    samples = np.random.choice(image_files, min(num_samples, len(image_files)), replace=False)
    
    # Create output directory
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Process each sample
    for i, img_path in enumerate(samples):
        # Run inference
        results = model(str(img_path))
        
        # Save the visualization
        result_plot = results[0].plot()
        cv2.imwrite(os.path.join(vis_dir, f'prediction_{i}.jpg'), result_plot)


def main():
    """Main function to execute the training and evaluation pipeline."""
    start_time = datetime.now()
    print(f"=== Starting YOLOv8 Segmentation Training Pipeline at {start_time} ===")
    
    # Step 1: Check dataset structure
    if not check_dataset_structure():
        print("Please fix the dataset structure and try again.")
        return
    
    # Step 2: Create or verify data.yaml
    data_yaml_path = create_data_yaml()
    
    # Step 3: Train the model
    model_path, training_results = train_model(data_yaml_path)
    print(f"Model saved at: {model_path}")
    
    # Step 4: Evaluate the model
    val_results, test_results = evaluate_model(model_path)
    
    # Step 5: Generate confusion matrix
    confusion_matrix_created = generate_confusion_matrix(model_path)
    
    # Step 6: Plot metrics
    plot_metrics(training_results)
    
    # Step 7: Visualize predictions
    visualize_predictions(model_path)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"=== YOLOv8 Segmentation Training Pipeline Completed ===")
    print(f"Total Duration: {duration}")
    print(f"Results saved to: {OUTPUT_DIR}")
    
    # Return performance summary
    print("\n=== Performance Summary ===")
    print(f"Validation mAP50: {val_results.box.map50:.4f}")
    print(f"Validation mAP50-95: {val_results.box.map:.4f}")
    print(f"Test mAP50: {test_results.box.map50:.4f}")
    print(f"Test mAP50-95: {test_results.box.map:.4f}")
    
    if hasattr(val_results, 'seg'):
        print(f"Segmentation Validation mAP50: {val_results.seg.map50:.4f}")
        print(f"Segmentation Validation mAP50-95: {val_results.seg.map:.4f}")
        print(f"Segmentation Test mAP50: {test_results.seg.map50:.4f}")
        print(f"Segmentation Test mAP50-95: {test_results.seg.map:.4f}")


if __name__ == "__main__":
    main()