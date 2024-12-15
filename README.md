# FASHION-MNIST-CLASSIFICATION
# **Fashion MNIST Image Classification**

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

## **Future Improvements**
- Experiment with different CNN architectures to improve accuracy.
- Add support for hyperparameter tuning using libraries like Keras Tuner.
- Incorporate advanced visualization techniques such as Grad-CAM for better interpretability.


