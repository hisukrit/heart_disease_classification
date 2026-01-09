# Heart Disease Classification using Machine Learning

---

## a. Problem Statement

Heart disease is one of the major health concerns worldwide, and early diagnosis can help reduce severe complications and mortality. With the availability of medical data, machine learning techniques can be used to assist in predicting the presence of heart disease in patients.

The aim of this project is to build multiple machine learning classification models to predict whether a patient has heart disease based on clinical features and to compare the performance of these models using standard evaluation metrics.

---

## b. Dataset Description

The dataset used for this project is the **Heart Disease Dataset** obtained from Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains medical records of patients along with a target variable indicating the presence or absence of heart disease.

- The dataset consists of more than 500 records.
- There are 13 input features such as age, sex, chest pain type, cholesterol level, resting blood pressure, and maximum heart rate.
- The target column `target` indicates:
  - `0` – No heart disease  
  - `1` – Presence of heart disease  

This dataset is suitable for a binary classification problem.

---

## c. Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| kNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

---

## d. Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Performs well with a high AUC score, but overall accuracy is lower compared to ensemble models |
| Decision Tree | Shows very high accuracy and MCC, indicating strong performance but with potential overfitting |
| kNN | Provides balanced results, though performance depends on distance calculations and feature scaling |
| Naive Bayes | Gives reasonable performance despite assuming independence among features |
| Random Forest (Ensemble) | Performs extremely well across all metrics due to ensemble learning |
| XGBoost (Ensemble) | Achieves perfect results, showing strong generalization and effectiveness of boosting |

---

## e. Conclusion

This project demonstrates the application of different machine learning classification techniques for predicting heart disease. Among all the models evaluated, ensemble methods such as **Random Forest** and **XGBoost** achieved the best performance. The results highlight the effectiveness of ensemble learning techniques when applied to medical datasets.

## f. Explore yourself

Link to Streamlit - https://heartdiseaseclassification-ml-assignment.streamlit.app/
