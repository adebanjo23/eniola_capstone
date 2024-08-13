# Intel Image Classification Using PyTorch

This project aims to build an image classification model using PyTorch that can accurately classify images into one of six predefined categories. The primary focus is on employing convolutional neural networks (CNNs) to achieve high accuracy in image classification. The dataset used is the Intel Image Classification Dataset, which contains a diverse set of images suitable for training and testing classification algorithms.

## Objectives

1. Data Preparation 
2. Data Augmentation and Loading
3. Exploratory Data Analysis
4. Model Development using Custom CNN
5. Model Summary and architecture
6. Initial Model Training and Validation
7. Initial Model Evaluation
8. Hyperparameter Tuning
9. Model Retraining with Hyperparameters
10. Final Model Evaluation
11. Visualization of model results
12. Model Saving and Loading

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- requests
- Pillow
- zipfile
- kaggle
- optuna
- Jupyter Notebook or JupyterLab

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/adebanjo23/eniola_capstone.git
    cd intel-image-classification
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

The dataset for this project is the Intel Image Classification Dataset, which can be found on Kaggle. It consists of natural scene images organized into six different categories:
1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

You can download the dataset from the following link: [Intel Image Classification Dataset](https://www.kaggle.com/puneet6060/intel-image-classification)

## Project Structure

- `intel_image_classification.ipynb`: Jupyter Notebook containing the complete project code and analysis.
- `application.log`: Log file capturing detailed insights of each step.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.

## Usage

1. **Run the Jupyter Notebook:**
    ```sh
    jupyter notebook intel_image_classification.ipynb
    ```

2. **Verify CUDA and GPU Availability:**
    Ensure that CUDA and GPU are properly set up and recognized by PyTorch.

3. **Data Preparation:**
    - Download and extract the dataset from Kaggle, you would need to setup your kaggle authentication.You can follow the steps [here](https://www.kaggle.com/docs/api)
    - Apply data augmentation techniques.
    - Split the dataset into training, validation, and testing sets.

4. **Exploratory Data Analysis (EDA):**
    - Visualize sample images and analyze class distribution.

5. **Model Development and Summary:**
    - Define a custom CNN model for image classification.

6. **Model Training and Validation:**
    - Train the model and validate it using the validation set.
    - Save the best model based on the validation loss.

7. **Model Evaluation:**
    - Evaluate the model on the test set.
    - Calculate metrics such as accuracy, precision, recall, and F1 score.
    - Plot confusion matrix and ROC curves.

8. **Hyperparameter Tuning:**
    - Perform hyperparameter tuning to optimize the model's performance.

9. **Model Retraining with best Hyperparameters:**
    - Train the model on the selected hyperparameters.

10. **Final Model Evaluation:**
     - Evaluate the model on the test set.

11. **Visualization of Results:**
    - Visualize the confusion matrix and class activation maps (CAM).

12. **Model Saving and Loading:**
    - Implement functionality to save and load the entire model architecture and weights.

## Logging

Detailed logging has been integrated into all steps to ensure that we have insights into everything happening during the execution of the notebook. Logs are saved in the `application.log` file.
