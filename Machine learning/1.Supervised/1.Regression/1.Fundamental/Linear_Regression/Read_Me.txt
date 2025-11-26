# Linear Regression: Study Hours vs Final Marks

## 📌 Overview
This project implements a Linear Regression model to predict students' final marks based on their attendance hours. The model demonstrates the relationship between study time and academic performance using real-world educational data.

---

## 📂 Project Structure
- **model.py** → Core implementation of Linear Regression (training + evaluation functions)
- **Linear_Regression.ipynb** → Complete Jupyter notebook with EDA, preprocessing, model training, and evaluation
- **Study_vs_Score_data.csv** → Dataset containing attendance hours and final marks
- **README.md** → Project documentation

---

## ⚙️ Workflow
1. **Data Loading & Exploration**
   - Import and examine the dataset
   - Perform exploratory data analysis (EDA)

2. **Data Preprocessing**
   - Split data into training and testing sets (80-20 split)
   - Apply StandardScaler for feature normalization

3. **Model Training**
   - Train Linear Regression model using scikit-learn
   - Fit the model on normalized training data

4. **Evaluation & Prediction**
   - Make predictions on test data
   - Evaluate performance using MSE and R² score
   - Visualize results and model performance

---

## 📊 Model Performance
- **Mean Squared Error (MSE)**: 21.43
- **R² Score**: 0.82

The model explains approximately 82% of the variance in final marks based on attendance hours.

---

## 🛠️ Requirements
```python
pandas
matplotlib
scikit-learn
numpy