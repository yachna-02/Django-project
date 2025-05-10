

# 🔮 Fetal Health Prediction

<p align="center">
  <img src="https://github.com/user-attachments/assets/b15d596b-eece-4c82-a9d8-ccd31aaebbee" alt="Fetal Health Prediction">
</p>

The **Fetal Health Prediction Project** leverages advanced **Machine Learning (ML)** and **Deep Learning** techniques to analyze **Cardiotocography (CTG)** data, aiming to provide accurate, real-time fetal health assessments. By classifying fetal health into **Normal**, **Suspect**, or **Pathological** categories, this project enhances prenatal care, reduces risks, and supports clinicians in making informed decisions.

---

## 📜 Table of Contents

1. Overview
2. Key Features 
3. Tech Stack 
4. Project Workflow
5. Algorithms and Techniques
6. Performance Metrics
7. Challenges and Solutions


---

## 🌟 Overview

Cardiotocography (CTG) is a crucial tool used in fetal monitoring during pregnancy. It measures **fetal heart rate (FHR)** and **uterine contractions (UC)**, which are vital indicators of fetal well-being. However, manual interpretation of CTG data can be inconsistent and prone to errors.  

This project introduces a machine learning-based system to automate and enhance fetal health assessment. It employs diverse algorithms, including traditional ML models like Logistic Regression and advanced architectures such as CNNs and LSTMs, to analyze CTG data efficiently.

### Objectives:
- Classify fetal health into **Normal**, **Suspect**, or **Pathological** categories.
- Assist clinicians with real-time, accurate predictions.
- Reduce errors and inconsistencies in manual CTG interpretation.
- Ensure timely medical interventions in critical cases.

---

## ✨ Key Features

1. **Automated Health Classification**:
   - Analyzes CTG data to predict fetal health status (Normal, Suspect, Pathological).
2. **Real-Time Predictions**:
   - Integrated models optimized for real-time monitoring with rapid response times.
3. **Multi-Model Approach**:
   - Combines interpretable models (Logistic Regression) with complex, high-performance models (CNNs, LSTMs).
4. **Data Imbalance Handling**:
   - Uses **SMOTE** to address imbalances in class distribution, ensuring the model effectively learns from rare pathological cases.
5. **Explainability**:
   - Includes interpretable algorithms like Logistic Regression to provide insights into predictions, ensuring trustworthiness in clinical use.

---

## 🛠 Tech Stack

- **Programming Language**: Python  
- **Libraries and Frameworks**:
  - **scikit-learn**: Data preprocessing, feature selection, traditional ML models.
  - **TensorFlow/Keras**: Deep Learning models (CNN, RNN, LSTM).
  - **LightGBM and CatBoost**: Gradient boosting algorithms.
  - **pandas** and **NumPy**: Data manipulation and analysis.
  - **Matplotlib** and **Seaborn**: Data visualization.

- **Techniques**:
  - SMOTE for handling class imbalance.
  - Grid Search and Random Search for hyperparameter tuning.
  - Dropout and Batch Normalization for reducing overfitting.

---

## 🔄 Project Workflow

1. **Data Collection**:
   - Gathered CTG data, including FHR, UC, and fetal movements, to form the dataset.

2. **Preprocessing**:
   - Removed missing values and outliers.
   - Selected relevant features using **SelectKBest**.
   - Applied scaling and normalization for distance-based models like KNN.

3. **Handling Class Imbalance**:
   - Addressed the imbalance between normal and pathological cases using **SMOTE** and ensemble techniques.

4. **Model Training**:
   - Trained various models, including Logistic Regression, SVC, KNN, LightGBM, CatBoost, CNN, RNN, and LSTM.

5. **Evaluation**:
   - Assessed models using multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

6. **Deployment**:
   - Integrated the best-performing models (LightGBM, LSTM) into a real-time prediction system.

---

## 🧠 Algorithms and Techniques

### Traditional Machine Learning Models:
1. **Logistic Regression**:
   - A simple, interpretable baseline model for binary and multi-class classification.
2. **Support Vector Classifier (SVC)**:
   - Handles non-linear relationships in data using kernel functions.
3. **K-Nearest Neighbors (KNN)**:
   - Classifies new cases based on the majority class of nearest neighbors.

### Gradient Boosting Models:
4. **LightGBM**:
   - Provides fast, real-time predictions with minimal computational cost.
5. **CatBoost**:
   - Efficiently handles categorical data without preprocessing and avoids overfitting.

### Deep Learning Models:
6. **1D Convolutional Neural Network (1D CNN)**:
   - Captures sequential patterns in CTG time-series data.
7. **Recurrent Neural Network (RNN)**:
   - Maintains memory of past inputs to capture temporal dependencies.
8. **Long Short-Term Memory (LSTM)**:
   - Solves vanishing gradient issues and captures long-term trends in sequential data.

---

## 📈 Performance Metrics

The models were evaluated on the following metrics:
1. **Accuracy**: Overall correctness of predictions.
2. **Precision**: Proportion of true positives among predicted positives.
3. **Recall (Sensitivity)**: Ability to detect true positive cases, prioritizing critical pathological predictions.
4. **F1-Score**: Harmonic mean of Precision and Recall.
5. **ROC-AUC Curve**: Measures the ability to distinguish between classes.

---

## 🛑 Challenges and Solutions

1. **Class Imbalance**:
   - **Problem**: Pathological cases were underrepresented.
   - **Solution**: Used SMOTE to generate synthetic samples for rare cases.

2. **Overfitting**:
   - **Problem**: Models performed well on training data but poorly on unseen data.
   - **Solution**: Applied Dropout, Batch Normalization, and Cross-Validation.

3. **Real-Time Constraints**:
   - **Problem**: Predictions needed to be instantaneous.
   - **Solution**: Used lightweight algorithms like LightGBM for faster inference.

4. **Interpretability**:
   - **Problem**: Clinicians need interpretable results.
   - **Solution**: Combined Logistic Regression with complex models for balance.



---

This project demonstrates the potential of combining machine learning and healthcare to improve maternal and fetal outcomes. It highlights how predictive systems can save lives by providing timely, accurate, and actionable insights. 🚀
