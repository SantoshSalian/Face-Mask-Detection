# Face-Mask-Detection

Introduction

This project aims to classify individuals based on their use of face masks into three categories:

1.Wearing a mask properly

2.Not wearing a mask

3.Wearing a mask improperly

The project was implemented using computer vision techniques and machine learning models. It leverages Convolutional Neural Networks (CNNs), feature extraction methods like HOG and SIFT, and classifiers such as Artificial Neural Networks (ANN) and Support Vector Machines (SVM). Additionally, the models were tested in real-time using video input.

Features

Image Classification: Detects the presence and correctness of face mask usage in images.

Real-Time Detection: Classifies individuals from a video stream in real-time.

Multiple Models:

CNN for end-to-end learning from raw image data.

HOG and SIFT with ANN for feature-based learning.

HOG and SIFT with SVM for feature-based classification.


Video Integration: Uses OpenCV to process video input and provide live feedback.

Technologies Used

Frameworks and Libraries:

TensorFlow, Keras: For building and training CNN and ANN models.

OpenCV: For real-time video feed integration and processing.

Scikit-Learn: For implementing SVM and feature-based classifiers.

NumPy, Pandas: For data manipulation and preprocessing.

Feature Extraction:

HOG (Histogram of Oriented Gradients)

SIFT (Scale-Invariant Feature Transform)


Results

Achieved classification using both end-to-end learning with CNNs and feature-based learning with HOG/SIFT and SVM/ANN.

Implemented real-time detection with minimal latency using pre-trained and custom-trained models.

Compared model performances using metrics like accuracy, precision, recall, F1-score, and AUC.
