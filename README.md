# Intelligent Breed Detection for Indian Cows and Buffaloes

A deep learning project for classifying Indian cow and buffalo breeds into 50 categories using Convolutional Neural Networks (CNN).

## Project Overview

This project implements a complete deep learning pipeline for image classification, specifically designed to identify and classify different breeds of Indian cows and buffaloes. The system uses a custom CNN architecture built with TensorFlow/Keras.

## Dataset Details

- **Total Breeds**: 50
- **Total Images**: Approximately 12,000–15,000
- **Image Type**: RGB
- **Image Formats**: .jpg, .jpeg, .png
- **Resolution Range**: 256×256 to 1024×1024
- **Annotations**: Breed Name, Region, Data Source

## Features

### 1. Data Preprocessing
- Automatic image resizing to 224×224
- Pixel normalization (0–1 scaling)
- Optional noise reduction
- Optional histogram equalization
- Automatic label encoding
- 80% training / 20% testing split

### 2. Data Augmentation
- Random rotation (±20°)
- Horizontal flip
- Random zoom (0.2)
- Brightness adjustment

### 3. Model Architecture
- 4 Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Dropout regularization (0.25–0.5)
- Fully connected Dense layers
- Softmax output layer with 50 classes

### 4. Training Configuration
- Optimizer: Adam
- Learning rate: 0.0001
- Loss Function: Categorical Crossentropy
- Batch size: 32
- Epochs: 15–20 (configurable)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau

### 5. Evaluation Metrics
- Accuracy
- Precision (weighted and per-class)
- Recall (weighted and per-class)
- F1 Score (weighted and per-class)
- Confusion Matrix

### 6. Visualization
- Training vs validation accuracy plots
- Training vs validation loss plots
- Confusion matrix heatmap

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset in the following structure:

```
data/
└── images/
    ├── breed_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── breed_2/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

Each breed should have its own folder containing images of that breed.

## Usage

### Training the Model

1. Update the `data_dir` path in `train_model.py` if your dataset is in a different location:
```python
CONFIG = {
    'data_dir': 'data/images',  # Update this path
    ...
}
```

2. Run the training script:
```bash
python train_model.py
```

The script will:
- Load and preprocess all images
- Split the dataset
- Build and train the CNN model
- Evaluate the model
- Generate visualizations
- Save the trained model and class mapping

### Making Predictions

Use the prediction script to classify new images:

```bash
python predict.py <image_path> [--top_k N]
```

Example:
```bash
python predict.py test_image.jpg
python predict.py test_image.jpg --top_k 10
```

Or use it programmatically:
```python
from predict import BreedPredictor

predictor = BreedPredictor(
    'models/cow_buffalo_breed_model.h5',
    'models/class_mapping.json'
)
results = predictor.predict('test_image.jpg', top_k=5)
```

## Output Files

After training, the following files will be generated:

- `models/cow_buffalo_breed_model.h5` - Trained model
- `models/class_mapping.json` - Mapping of class indices to breed names
- `results/training_history.png` - Training/validation accuracy and loss plots
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/metrics.json` - Evaluation metrics summary
- `results/prediction_result.png` - Prediction visualization (when using predict.py)

## Model Architecture Details

The CNN model consists of:

1. **Convolutional Block 1**: 32 filters, 3×3 kernel, MaxPooling, Dropout(0.25)
2. **Convolutional Block 2**: 64 filters, 3×3 kernel, MaxPooling, Dropout(0.25)
3. **Convolutional Block 3**: 128 filters, 3×3 kernel, MaxPooling, Dropout(0.25)
4. **Convolutional Block 4**: 256 filters, 3×3 kernel, MaxPooling, Dropout(0.25)
5. **Dense Layers**: 512 → 256 neurons with Dropout(0.5)
6. **Output Layer**: 50 classes with Softmax activation

Total parameters: ~15-20 million (depending on input size)

## Configuration

You can modify the training configuration in `train_model.py`:

```python
CONFIG = {
    'image_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.0001,
    'validation_split': 0.2,
    'test_split': 0.2,
    'num_classes': 50,
    'data_dir': 'data/images',
    'model_save_path': 'models/cow_buffalo_breed_model.h5',
    'class_mapping_path': 'models/class_mapping.json',
    'results_dir': 'results'
}
```

## Results Interpretation

The model evaluation provides comprehensive metrics that can be used to:
- Compare with other architectures (VGG16, ResNet, EfficientNet)
- Identify breeds that are difficult to classify
- Understand model performance across different classes
- Guide further model improvements

Key metrics to compare:
- **Overall Accuracy**: General classification performance
- **Weighted F1 Score**: Balanced performance across all classes
- **Per-class Precision/Recall**: Identify problematic breeds
- **Confusion Matrix**: Visualize misclassification patterns

## Future Enhancements

This CNN model serves as a baseline for comparison with:
- Transfer learning models (VGG16, ResNet50, EfficientNet)
- Advanced architectures (Vision Transformers)
- Ensemble methods
- Data augmentation improvements

## Troubleshooting

### No images found
- Ensure images are in the correct directory structure
- Check that image formats are .jpg, .jpeg, or .png
- Verify the `data_dir` path in CONFIG

### Out of memory errors
- Reduce batch size (e.g., 16 or 8)
- Reduce image size (e.g., 128×128)
- Use fewer epochs

### Poor accuracy
- Increase dataset size
- Add more data augmentation
- Adjust learning rate
- Increase model capacity
- Check for class imbalance

## License

This project is for educational and research purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.


















