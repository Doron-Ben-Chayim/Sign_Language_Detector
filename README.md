# **Sign Language Detector**

This repository aims to **train models** capable of making **real-time predictions** of sign language letters based on input from a **camera**. The project leverages **two datasets** and **three models** to improve accuracy and performance. It emphasizes the importance of **feature engineering** for achieving reliable predictions.

---

## **Overview**

The main goal of this project is to enable **real-time detection of sign language letters** from a video feed using machine learning models. With **input from a camera**, the trained models can predict what letter the user is signing. The repository demonstrates the full lifecycle of building such a system, including **data preprocessing, feature extraction, model training**, and **real-time inference**.

---

## **Datasets Used**

1. **MNIST Sign Language Dataset**  
   - Contains images of hand signs representing letters from A-Y (No J and Z).  
   - Used primarily for training a **CNN model** and **XGBoost model**.

2. **ASL Sign Language Dataset**  
   - A custom dataset with various hand signs in different orientations and lighting conditions.  
   - Used for **XGBoost and FFNN models** with feature engineering techniques.

---

## **Models Used**

1. **Convolutional Neural Network (CNN)**  
   - Trained on the **MNIST Sign Language dataset** for image-based classification.  
   - Helps in understanding the effectiveness of traditional CNNs in predicting static signs.

2. **Feedforward Neural Network (FFNN)**  
   - Trained on the **ASL dataset** using **Mediapipe-generated features** as input.  
   - Demonstrates how simple architectures can perform well with effective feature extraction.

3. **XGBoost Model**  
   - Uses **Mediapipe-generated features** for training.  
   - Provides fast and reliable predictions in real-time through feature engineering.

---

## **Feature Engineering and Mediapipe Integration**

- **Mediapipe**, a powerful library for real-time human pose detection, is used to generate **hand landmarks** from video frames.
- These landmarks serve as **input features** for both the **FFNN** and **XGBoost models**.
- The project demonstrates the **importance of feature engineering** by showing how these models can outperform raw image-based predictions in real-time.

---

## **Project Workflow**

1. **Data Collection and Preprocessing**  
   - Utilize both the **MNIST** and **ASL datasets** for training.  
   - Preprocess data by resizing images and normalizing pixel values.

2. **Feature Generation**  
   - Use **Mediapipe** to extract key landmarks from video frames.  
   - Convert landmarks into meaningful features for FFNN and XGBoost models.

3. **Model Training**  
   - Train a **CNN** using the MNIST dataset.  
   - Train an **FFNN** and **XGBoost** model using the extracted Mediapipe features from the ASL dataset.

4. **Real-Time Inference**  
   - The **camera input** provides a stream of video frames.  
   - The trained models predict the **letter being signed** in real-time.

---
## **Data**

The data such as parquets of the features used can be found in this google drive:
https://drive.google.com/drive/folders/1hYmc8fmFdoNagUPByilGwVaeJp9ifKzh?usp=sharing

The Origional Datasets can be found at the following websites:
   - https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data
   - https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data
