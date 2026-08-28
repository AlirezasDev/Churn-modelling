# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Machine Learning pipeline for predicting customer churn using multiple classification models.

## Live Demo

Run the Streamlit app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Overview

This project implements a complete machine learning pipeline to predict whether a customer will churn (leave the service) based on their demographic and financial attributes. The project follows industry best practices for data science workflows, including data preprocessing, feature engineering, model training, evaluation, and deployment.

## Features

### Notebook
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Encoding categorical variables, feature scaling
- **Multiple Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Model Evaluation**: Accuracy, Precision, Recall, F1, ROC AUC metrics
- **ROC Analysis**: Comparison of ROC curves across models
- **Production Ready**: Saved models and scaler for deployment

### Streamlit Web App
- **Interactive Dashboard**: 6 tabs with different visualizations
- **Real-time Predictions**: Watch model training and predictions
- **Model Comparison**: Compare multiple ML models side-by-side
- **Feature Importance**: Understand what drives churn predictions
- **Live Prediction**: Input customer details and get instant churn prediction
- **Gauge Chart**: Visual probability indicator
- **Dark Theme**: Modern, attractive UI with gradient backgrounds
- **Responsive Design**: Works on desktop and mobile

## Project Structure

```
Churn-modelling/
├── Project2.ipynb              # Main notebook with complete pipeline
├── app.py                      # Streamlit web application
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── Churn.csv                   # Customer churn dataset
├── requirements.txt            # Python dependencies
├── best_churn_model.pkl        # Trained model (generated)
├── scaler.pkl                  # Feature scaler (generated)
└── README.md                   # This file
```

## Dataset

- **Source**: Bank customer data
- **Records**: 10,000 customers
- **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target**: Exited (1 = Churned, 0 = Stayed)

## Methodology

### 1. Data Loading & Preprocessing
- Load and inspect dataset
- Drop non-informative columns (RowNumber, CustomerId, Surname)
- Handle missing values (if any)

### 2. Feature Engineering
- **Label Encoding**: Convert Gender to numeric
- **One-Hot Encoding**: Convert Geography to dummy variables
- **Feature Scaling**: StandardScaler for numerical features

### 3. Model Training
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential ensemble method
- **XGBoost**: Regularized gradient boosting
- **LightGBM**: Fast gradient boosting framework
- **Cross-Validation**: 5-fold CV for model comparison

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

## Installation

```bash
# Clone the repository
git clone https://github.com/AlirezasDev/Churn-modelling.git
cd Churn-modelling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebook
```bash
jupyter notebook Project2.ipynb
```

### Running the Web App
```bash
streamlit run app.py
```

### Using the Saved Model
```python
import joblib
import pandas as pd

model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare your features
X_new = scaler.transform(your_features)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]
```

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|----|----|
| Logistic Regression | - | - | - | - | - |
| Random Forest | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - |
| XGBoost | - | - | - | - | - |
| LightGBM | - | - | - | - | - |

*Results will be populated after running the notebook*

## Technologies Used

- Python 3.8+
- Pandas & NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib & Seaborn
- Streamlit
- Plotly
- Jupyter Notebook

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Scikit-learn](https://scikit-learn.org/) documentation
- [XGBoost](https://xgboost.readthedocs.io/) documentation
- [LightGBM](https://lightgbm.readthedocs.io/) documentation
- [Streamlit](https://streamlit.io/) documentation

---

**Author**: Alireza Sepehri  
**Email**: alireza_sepehri@mathdep.iust.ac.ir  
**GitHub**: [@AlirezasDev](https://github.com/AlirezasDev)