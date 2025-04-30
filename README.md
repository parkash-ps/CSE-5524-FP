# Automated Cardiac Parameter Extraction from M-mode Echocardiography

This repository contains the implementation of a deep learning-based system for automating the extraction of cardiac parameters from M-mode echocardiography images. The project uses various segmentation models to precisely delineate cardiac walls and automate the measurement of clinically relevant cardiac parameters.

## Project Overview

Manual analysis of echocardiographic traces is time-consuming, requires significant expertise, and introduces observer variability. This project aims to develop an automated system that can:

1. Accurately segment cardiac walls in M-mode echocardiography images
2. Extract key clinical parameters including left ventricular dimensions and wall thicknesses
3. Provide consistent measurements that correspond with expert annotations

The system was developed using a dataset of 1500 echocardiographic images, including 70 annotated frames and 25 videos from murine models.

## Repository Contents

- `Medsam.ipynb`: Implementation of the Medical Segment Anything Model (MedSAM) for cardiac segmentation
- `ultrasound_processor.py`: Utility script for preprocessing ultrasound images and extracting regions of interest
- `yollov8.py`: Implementation and training script for YOLOv8 segmentation model

## Models

The repository implements and compares two main approaches:

1. **MedSAM (Baseline)**: Adaptation of the Segment Anything Model (SAM) for medical imaging applications
2. **YOLOv8 (Advanced)**: Fine-tuned YOLOv8 segmentation models for precise cardiac wall detection

Our experiments showed that YOLOv8 demonstrated superior performance in cardiac wall detection despite challenges of limited training data and image contrast variability.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Matplotlib
- ultralytics (for YOLOv8)
- segment_anything (for MedSAM)

## Usage

### MedSAM Implementation

```python
# Example usage of MedSAM for echo image segmentation
from Medsam import MedSAMModel, EchoImageProcessor

# Initialize the model
model = MedSAMModel('path/to/checkpoint')

# Initialize processor
processor = EchoImageProcessor(model)

# Load and process an image
image = processor.load_image('path/to/echo_image.png')

# Segment with a bounding box (can be defined interactively or automatically)
box = [x_min, y_min, x_max, y_max]  # Example values
mask, score = processor.segment_with_box(box)

# Save results
processor.save_results('output_dir', 'result_name')
```

### Ultrasound Processing

```python
# Extract ROI from an ultrasound image
from ultrasound_processor import extract_ultrasound_roi

# Process a single image
result = extract_ultrasound_roi('path/to/image.tif', 'output_path.png', save_visualization=True)

# Process a folder of images
from ultrasound_processor import process_folder
process_folder('input_folder', 'output_folder', save_visualization=True)
```

### YOLOv8 Training and Inference

```python
# To train the YOLOv8 model
# Ensure your dataset is properly formatted in YOLOv8-compatible structure
python yollov8.py

# The script will:
# 1. Check dataset structure
# 2. Create/verify data.yaml
# 3. Train the model
# 4. Evaluate on validation and test sets
# 5. Generate confusion matrix and visualizations
```

## Future Work

Planned enhancements include:

1. Parameter extraction implementation to extract specific cardiac measures
2. Clinical validation against VevoLab measurements
3. Model improvements through expanded annotated datasets
4. Evaluation of alternative architectures
5. Integration into cardiovascular research workflows



## License

MIT License
