# Skin-Cancer-Detector

Skin-Cancer-Detector is a deep learning project that uses a Convolutional Neural Network (CNN) to classify skin lesions into seven categories based on dermatoscopic images. The project leverages the HAM10000 dataset to train a model for early detection and diagnosis of skin cancer.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
- [Results](#results)
- [References](#references)

## Overview
Skin cancer is among the most common types of cancer globally, and early detection is vital for effective treatment. Skin-Cancer-Detector leverages CNNs to classify images of skin lesions into the following seven categories:
1. Actinic keratoses and intraepithelial carcinoma (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanocytic nevi (nv)
6. Vascular lesions (vasc)
7. Melanoma (mel)

The model is trained on the HAM10000 dataset, which includes 10,015 high-resolution dermatoscopic images.

## Dataset
The **HAM10000** dataset ("Human Against Machine with 10000 training images") is a diverse collection of dermatoscopic images of pigmented skin lesions. It includes:
- **Images**: High-resolution dermatoscopic images.
- **Metadata**: Detailed information for each image, including diagnosis.

## Model Architecture
The CNN model used in this project includes:
- **Input Layer**: Accepts images of size (28, 28, 3).
- **Convolutional Layers**: Extract features using filters and ReLU activation.
- **Pooling Layers**: Reduce spatial dimensions with MaxPooling.
- **Fully Connected Layers**: Learn nonlinear feature combinations.
- **Output Layer**: Softmax activation with 7 units for the 7 classes.

## Requirements
To run this project, ensure you have the following installed:
- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn (imblearn)
- Matplotlib
- Seaborn
- Pillow (PIL)
- tqdm

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HassanAsim/Skin-Cancer-Detector.git
   cd Skin-Cancer-Detector
   
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate

3. Install the required packages:
   ```bash
   pip install -r requirements.txt

## Usage

### Training
To train the CNN model:

1. **Download the HAM10000 dataset**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
   - Place the dataset files in the appropriate directories:
     - Images: `HAM10000_images_part_1/` and `HAM10000_images_part_2/`
     - Metadata CSV file: `HAM10000_metadata.csv`
     - Pixel values CSV file: `hmnist_28_28_RGB.csv`

2. **Open the notebook** `Skin_Cancer_Detector.ipynb`:

3. **Run the Training Code Cell**:
   - **Cell 1: Training Code**  
     This cell includes all necessary imports and the entire code required for training the CNN model.

     #### Key Operations:
     1. **Data Loading and Preprocessing**:
        - Reads the metadata and image pixel values.
        - Splits the data into features (`x`) and labels (`y`).
        - Handles class imbalance using `RandomOverSampler`.
        - Reshapes the data to the required input dimensions `(28, 28, 3)`.
        - Standardizes the data using the calculated mean and standard deviation.

     2. **Data Splitting**:
        - Splits the resampled data into training and testing sets using `train_test_split`.

     3. **Model Building**:
        - Constructs a CNN model using Keras Sequential API.
        - Defines the architecture with convolutional layers, pooling layers, and dense layers.

     4. **Model Compilation**:
        - Compiles the model with `sparse_categorical_crossentropy` loss, `adam` optimizer, and accuracy metric.

     5. **Model Training**:
        - Trains the model on the training data.
        - Uses `ModelCheckpoint` to save the best model during training.
        - Saves the final model as `final_model.h5`.

     #### To Run:
     - Execute this cell and wait for the training to complete.
     - Monitor the output for training progress and validation accuracy.

---

### Testing

Run the Testing Code Cell:

**Cell 2: Testing Code**  
This cell contains the code to evaluate the trained model.

#### Key Operations:

1. **Model and Parameter Loading**:
   - Loads the trained model (`final_model.h5`).
   - Loads the mean and standard deviation used for standardization.

2. **Data Loading**:
   - Reads the metadata CSV file.
   - Maps image IDs to their file paths.
   - Associates true labels with their corresponding indices.

3. **Prediction and Evaluation**:
   - Iterates over each image in the dataset.
   - Preprocesses images using the same standardization as in training.
   - Predicts the class of each image using the trained model.
   - Compares predicted labels with true labels.
   - Records the results and calculates the overall accuracy.

4. **Results Saving**:
   - Saves the detailed prediction results to `prediction_results.csv`.

#### To Run:
- Execute this cell after the training cell has completed.
- The cell will display a progress bar and print out the final accuracy.

---

## Results
- **Accuracy**: The model achieves an accuracy of approximately 94% on the test dataset.
- **Prediction Results**: The file `prediction_results.csv` contains:
  - Image IDs
  - True labels
  - Predicted labels
  - Confidence scores
- **Confusion Matrix**: You can use the results in `prediction_results.csv` to create a confusion matrix, providing insights into the classification performance for each category.

## References
- [HAM10000 Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

