# Plant Disease Detection Project 🌱📊

Welcome to the **Plant Disease Detection Project**! This repository demonstrates the implementation of a deep learning solution for detecting plant diseases from leaf images using the ResNet101 architecture. The entire project is designed to run seamlessly on Google Colab, leveraging its GPU capabilities for efficient training and inference.

## Project Overview
Identifying plant diseases at an early stage is crucial for improving agricultural productivity and reducing crop losses. In this project, we utilize a pre-trained ResNet101 model, fine-tuned on a dataset of diseased and healthy plant leaf images, to classify plant health conditions effectively.

## Key Features
- **Pre-Trained Model**: Utilizes ResNet101 for transfer learning to ensure high accuracy with fewer computational resources.
- **Google Colab Integration**: Provides a cloud-based environment with GPU acceleration for training and testing.
- **Comprehensive Workflow**: Covers data preprocessing, model training, evaluation, and predictions.

## Dataset
The dataset contains high-quality images of plant leaves categorized into:
- **Healthy Leaves**
- **Diseased Leaves** (various types of diseases)

Each image is labeled with its respective plant type and health status.

### Dataset Attributes
- High-resolution images of plant leaves.
- Multiple classes representing different diseases and healthy leaves.
- Data is preprocessed into training, validation, and testing sets.

## Tools and Libraries
This project leverages the following tools and libraries:
- **Python**: Primary programming language.
- **TensorFlow/Keras**: Framework for deep learning model development.
- **OpenCV**: For image preprocessing.
- **Matplotlib & Seaborn**: For visualizing results.
- **Google Colab**: Cloud platform for running the project.

## Project Workflow
1. **Setup**:
   - Import necessary libraries.
   - Mount Google Drive for dataset access.
2. **Data Preprocessing**:
   - Resize images to 224x224 for compatibility with ResNet101.
   - Normalize pixel values.
   - Split data into training, validation, and testing sets.
3. **Model Development**:
   - Load the pre-trained ResNet101 model with ImageNet weights.
   - Add custom layers for plant disease classification.
   - Compile the model with suitable optimizer and loss function.
4. **Training**:
   - Train the model on the processed dataset.
   - Use Google Colab’s GPU for faster training.
5. **Evaluation**:
   - Evaluate model performance using metrics like accuracy and F1-score.
   - Visualize training and validation loss/accuracy curves.
6. **Prediction**:
   - Test the model on unseen leaf images.
   - Display predictions with confidence scores.

## How to Use
1. Open the Google Colab notebook:
   - [Link to Notebook](https://colab.research.google.com/drive/12HdlHq5UT9i7QSSUS5szcNdqvze9peKs#scrollTo=JvMksvFXluz6)
2. Mount Google Drive to access the dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install dependencies:
   ```python
   !pip install tensorflow opencv-python matplotlib
   ```
4. Run the notebook cells sequentially to train and evaluate the model.
5. Use the provided `predict` function to classify new leaf images.

## Results
- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~94%

## Visualizations
- **Training Curves**: Visualize accuracy and loss across epochs.
- **Confusion Matrix**: Analyze model performance on each class.
- **Sample Predictions**: Display leaf images with predicted labels and confidence scores.

## Acknowledgments
- **Dataset Source**: Acknowledgment to the publicly available plant disease datasets.
- **ResNet101**: Thanks to the creators of ResNet for providing a powerful pre-trained architecture.
- **Google Colab**: For offering free GPU resources for model training.

---

Feel free to contribute to this project or raise any issues. Let’s work together to build smarter agricultural solutions!

