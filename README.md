# Brain_Tumor_Classification
This is a deep learning-based project for brain tumor classification using a Convolutional Neural Network (CNN). The model classifies brain MRI images into four categories:

Glioma Tumor
Meningioma Tumor
No Tumor
Pituitary Tumor
The model achieves an accuracy of 95% in classifying the brain tumor images based on the dataset of brain MRI scans.

Features
Image Classification: Classifies brain MRI scans into one of the four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor.
High Accuracy: Achieved a classification accuracy of 95% on a test dataset.
Model Architecture: Built using a CNN architecture with multiple convolutional layers, max pooling, dropout, and dense layers for classification.
Preprocessing: Applied image resizing, normalization, and augmentation techniques to improve model performance.
Tech Stack
Deep Learning Framework:

TensorFlow / Keras for building and training the CNN model.
OpenCV and PIL for image preprocessing.
Programming Language:

Python for writing the entire codebase.
Libraries:

NumPy and Pandas for data handling.
Matplotlib and Seaborn for data visualization.
Scikit-learn for performance evaluation (e.g., accuracy, confusion matrix).
Dataset
The model was trained on the Brain MRI Images dataset available from various sources like Kaggle.
The dataset contains MRI scans labeled with categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor.
Model Architecture
The CNN model architecture includes:

Convolutional Layers: Extracts features from the MRI images using multiple convolutional layers.
Max-Pooling Layers: Reduces the spatial dimensions of the image.
Dropout Layers: Prevents overfitting by randomly dropping a fraction of the units during training.
Fully Connected Layers: Classifies the extracted features into the target classes.
Softmax Activation: For multi-class classification (4 classes).
Results
Accuracy: 95% on the test set.
Confusion Matrix: Provides insights into how well the model is performing across different classes.
Loss: Achieved low loss after several epochs of training.
Installation
Prerequisites:
Python 3.x installed.
TensorFlow, Keras, and other required libraries.
