import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def extract_ultrasound_roi(image_path, output_path=None, save_visualization=False):
    """
    Extract the main ROI from an ultrasound image while preserving coordinate information
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the ROI image (if None, ROI is not saved)
        save_visualization: Whether to save the visualization of original and ROI
        
    Returns:
        Dictionary containing ROI information
    """
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to detect the ultrasound area (dark background, bright content)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour (likely the ultrasound area)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        # Extract ROI
        roi = img[y:y+h, x:x+w]
        
        # For demonstration, using placeholder values for x_coords, y_coords, x_scale, y_scale
        # These would normally be extracted from the ultrasound image metadata or scale markers
        x_coords = [0, w]
        y_coords = [0, h]
        x_scale = 1.0  # seconds
        y_scale = 1.0  # cm

        # Save ROI if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(str(output_path), roi)

        # Visualize and save if requested
        if save_visualization:
            vis_path = str(output_path).replace('.png', '_visualization.png') if output_path else None
            plt.figure(figsize=(10, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], 'r-', linewidth=2)
            plt.title('Original Image with ROI')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            plt.title('Extracted ROI')

            plt.tight_layout()
            
            if vis_path:
                plt.savefig(vis_path)
                plt.close()
            else:
                plt.show()

        return {
            'roi': roi,
            'roi_gray': cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
            'x_start': x,
            'y_start': y,
            'width': w,
            'height': h,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'x_scale': x_scale,  # seconds
            'y_scale': y_scale   # cm
        }
    else:
        raise ValueError(f"No contours found in the image {image_path}")

def process_folder(input_folder, output_folder, save_visualization=False):
    """
    Process all TIFF images in a folder and save the ROIs as PNG to another folder
    
    Args:
        input_folder: Path to the folder containing TIFF images
        output_folder: Path to the folder to save the processed images as PNG
        save_visualization: Whether to save visualizations alongside ROIs
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all TIFF files
    input_path = Path(input_folder)
    tiff_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
    
    if not tiff_files:
        print(f"No TIFF files found in {input_folder}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process")
    
    # Process each file
    results = []
    for img_path in tiff_files:
        try:
            # Determine output path (with .png extension)
            output_filename = img_path.stem + '.png'
            output_path = Path(output_folder) / output_filename
            
            print(f"Processing {img_path}...")
            result = extract_ultrasound_roi(img_path, output_path, save_visualization)
            results.append(result)
            print(f"Saved ROI to {output_path}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process ultrasound TIFF images and extract ROIs as PNG")
    parser.add_argument("input_folder", help="Folder containing TIFF images")
    parser.add_argument("output_folder", help="Folder to save processed images as PNG")
    parser.add_argument("--visualize", action="store_true", help="Save visualizations of original and ROI")
    args = parser.parse_args()
    
    # Process the folder
    process_folder(args.input_folder, args.output_folder, args.visualize)
    print("Processing complete.")