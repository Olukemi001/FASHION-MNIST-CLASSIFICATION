# FASHION-MNIST-CLASSIFICATION IN BOTH PYTHON AND R
# **Fashion MNIST Image Classification in Python**

## **Project Overview**
This project focuses on classifying images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN). The dataset consists of 60,000 training images and 10,000 test images representing 10 classes of fashion items. The project is organized into five Python modules for efficient processing, model training, visualization, and integration.

---

## **Modules Overview**

### 1. **`data_processing.ipynb`**
   - Handles data loading and preprocessing.
   - Normalizes image pixel values and reshapes them for the CNN.
   - Converts labels into one-hot encoded format.

### 2. **`model_training.ipynb`**
   - Defines the CNN architecture.
   - Trains the model on the Fashion MNIST dataset.
   - Saves the trained model for reuse.

### 3. **`visualization.ipynb`**
   - Visualizes predictions by the trained model.
   - Displays test images with their predicted and actual labels.
   - Includes confidence scores for each prediction.

### 4. **`convert_ipynb_to_py.ipynb`**
   - Converts Jupyter Notebook files (`.ipynb`) into Python scripts (`.py`).
   - Ensures seamless execution of the project in script format.

### 5. **`main.ipynb`**
   - Orchestrates the entire workflow by integrating all other modules.
   - Executes data preprocessing, model training, evaluation, and visualization in sequence.

---

## **File Structure**

```
Fashion-MNIST-Project/
├── data_processing.ipynb       # Data preprocessing module
├── model_training.ipynb        # Model training module
├── visualization.ipynb         # Prediction and visualization module
├── convert_ipynb_to_py1.ipynb   # Notebook-to-script conversion module
├── main.ipynb                  # Main script integrating all modules
├── README.md                # Project documentation
```

---

## **Installation**

### **Dependencies**
The following libraries are required:
- Python 3.8 or higher
- TensorFlow
- NumPy
- Matplotlib
- Jupyter Notebook 

### **Steps to Set Up the Environment**

1. **Install Python**  
   Ensure Python 3.8 or higher is installed.

3. **Install Required Libraries**  
   ```bash

   pip install tensorflow keras numpy matplotlib --default-timeout=100

   ```

4. **Clone the Repository**  
   Clone the repository and run the modules

---

## **Usage**

### **1. Convert Notebooks to Python Scripts**
For using Jupyter Notebook files, convert them to Python scripts to prevent errors


### **2. Run the Main Script**
To execute the project end-to-end.

This will preprocess the data, train the model, evaluate it, and visualize predictions.



---

## **How It Works**

### **Data Preprocessing (`data_processing.py`)**
1. Loads the Fashion MNIST dataset.
2. Normalizes images by scaling pixel values to [0, 1, 2, 3].
3. Reshapes images to a format suitable for CNNs.
4. Converts integer labels into one-hot encoded vectors.

### **Model Training (`model_training.py`)**
1. Defines a CNN with:
   - Convolutional layers for feature extraction.
   - Pooling layers for dimensionality reduction.
   - Fully connected layers for classification.
   - Dropout for regularization.
2. Trains the model using the Adam optimizer and categorical crossentropy loss.
3. Saves the trained model for future use.

### **Visualization (`visualization.py`)**
1. Uses the trained model to predict classes of test images.
2. Displays test images alongside their predicted and actual labels.
3. Includes confidence scores for predictions.

### **Convert Notebooks (`convert_ipynb_to_py.py`)**
1. Converts `.ipynb` files to `.py` scripts for easier integration and execution.
2. Streamlines the transition between development in Jupyter Notebook and deployment.

### **Main Script (`main.py`)**
1. Calls the `data_processing` module to prepare the dataset.
2. Invokes the `model_training` module to train the CNN.
3. Uses the `visualization` module to showcase predictions.
4. Ensures smooth integration of all components.



## **Notes**
- Adjust the parameters (e.g., epochs, batch size) in `model_training.py` to experiment with different training configurations.
- Use the `visualization.py` module to explore model predictions on specific test samples.
- Ensure that dependencies are installed in the same environment to avoid import errors.

---

## **Visualisation Interpretation**
- Ankle boot with a confidence level of 99.98% was predicted.
- A pullover with a confidence level of 99.96% was predicted.
- A trouser with  a confidence level of 100% was predicted.
- Please not that the images were color graded from greyscale for better clarity.


# **Fashion MNIST Image Classification in R**

This repository contains an R script to train a Convolutional Neural Network (CNN) model using the Fashion MNIST dataset, a collection of 60,000 28x28 grayscale images of 10 fashion categories. The script demonstrates how to preprocess the data, build a CNN model, train it, evaluate performance, and visualize predictions using R and the `keras` package.

## Prerequisites

Before running the script, ensure that the following software and packages are installed:

### Software:
- R (version 4.4.2 or higher recommended)
- Python (Python 3.7, 3.8, or 3.9 are usually recommended) TensorFlow and Keras in R rely on the underlying Python environment.
- TensorFlow does not support Python 3.12 yet
- RStudio 

### Required R Packages:
- `keras`: For building and training neural network models
- `tensorflow`: Backend for Keras, responsible for performing computations
- `tensorflow` needs to be installed and configured properly for Keras to work. The function `install_keras()` handles the installation.

### Installation Instructions

Follow these steps to install the required libraries:

1. **Install Keras and TensorFlow in R:**

   The script automatically installs `keras` and `tensorflow` packages if not already installed using the `install_keras()` function.

   Alternatively, if you want to manually install the packages, run the following commands in R:

   ```r
   install.packages("keras")
   library(keras)
   install_keras()  # Installs TensorFlow and other dependencies
   ```

2. **Install additional dependencies**:
   If needed, run the following commands to install any other dependencies that the script uses.

   ```r
   install.packages("tensorflow")
   ```

## Script Breakdown

The script performs the following tasks:

### 1. **Install and Load Libraries**
   The `install_keras()` function ensures that the `keras` and `tensorflow` packages are installed and loaded correctly.

### 2. **Load and Preprocess the Fashion MNIST Dataset**
   The `load_and_preprocess_data()` function loads the Fashion MNIST dataset using the `dataset_fashion_mnist()` function provided by `keras`. It performs the following steps:
   - Normalizes the image data (scaling pixel values to the range [0, 1]).
   - Reshapes the images to have a channel dimension (28x28x1).
   - One-hot encodes the labels (converts categorical labels into binary vectors).

### 3. **Create the CNN Model**
   The `create_model()` function defines a Convolutional Neural Network (CNN) model with the following architecture:
   - **Conv2D Layer**: 32 filters with a kernel size of 3x3, ReLU activation.
   - **MaxPooling2D Layer**: Pool size of 2x2.
   - **Conv2D Layer**: 64 filters with a kernel size of 3x3, ReLU activation.
   - **MaxPooling2D Layer**: Pool size of 2x2.
   - **Flatten Layer**: Flatten the 2D images into 1D vectors.
   - **Dense Layer**: Fully connected layer with 128 units and ReLU activation.
   - **Dropout Layer**: Dropout with rate 0.5 to reduce overfitting.
   - **Dense Layer**: Output layer with 10 units (one for each class) and softmax activation to produce class probabilities.

   The model is compiled using the **Adam optimizer** and **categorical crossentropy loss**.

### 4. **Train the Model**
   The `train_model()` function trains the CNN model using the training data. It accepts the following parameters:
   - `epochs`: The number of epochs for training (default is 10).
   - `batch_size`: The size of each batch of data (default is 64).
   - `validation_split`: The proportion of the training data to use for validation (default is 0.2).

   The model is trained using the `fit()` method, and the training history is returned.

### 5. **Evaluate the Model**
   The `evaluate_model()` function evaluates the trained model on the test data. It returns the loss and accuracy on the test dataset.

### 6. **Visualize Predictions**
   The `visualize_predictions()` function visualizes the predictions made by the model for a given set of sample indices. It:
   - Takes a set of test images.
   - Makes predictions using the model.
   - Plots the true labels, predicted labels, and the confidence of the predictions.

### 7. **Class Names**
   The script includes the following class names corresponding to the Fashion MNIST dataset:
   ```r
   class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
   ```

### Example Workflow:
```r
# Load and preprocess data
data <- load_and_preprocess_data()

# Create CNN model
model <- create_model()

# Train the model
history <- train_model(model, data$x_train, data$y_train, epochs = 10)

# Evaluate the model
evaluation <- evaluate_model(model, data$x_test, data$y_test)
cat("Test Accuracy:", evaluation[2] * 100, "%\n")

# Visualize predictions
sample_indices <- c(1, 2, 3, 4)
visualize_predictions(model, data$x_test, data$y_test, sample_indices, class_names)
```

## Expected Output:
1. **Training History**: The `train_model()` function will print training progress including loss and accuracy per epoch.
2. **Test Accuracy**: The `evaluate_model()` function will output the test accuracy of the model after evaluation.
3. **Prediction Visualization**: The `visualize_predictions()` function will display the images along with the true labels, predicted labels, and prediction confidence for a subset of test samples.

## Troubleshooting
- **Error: Unable to install packages**: Ensure that you have internet access and the correct permissions to install packages. Running RStudio as an administrator may help resolve permission issues.
- **Error: No GPU available**: If you don't have a GPU available for TensorFlow, the model will still run on the CPU but will be slower. Make sure you have installed the necessary CPU version of TensorFlow.
- **Memory errors**: If you run into memory issues, try reducing the batch size or using a smaller subset of the dataset.


# **PROJECT CONCLUSION**
This project uses a Convolutional Neural Network (CNN) model built in both Python and R with the `keras` package to classify images from the Fashion MNIST dataset, which contains 60,000 28x28 grayscale images of 10 different fashion categories. The model architecture consists of two convolutional layers, max-pooling layers, and dense layers with dropout for regularization. After training the model for 10 epochs, the test accuracy is evaluated and predictions are visualized for a set of test images. The results show a well-trained model capable of making accurate predictions, with test accuracy around 90%, demonstrating the effectiveness of CNNs for image classification tasks.




