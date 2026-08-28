import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="c",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #888 !important;
    }
    .stMetric div {
        color: #fff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    div[data-testid="stForm"] {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('Churn.csv')
    return df


def preprocess_data(df):
    df_processed = df.copy()
    
    df_processed = df_processed.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    le_gender = LabelEncoder()
    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
    
    df_processed = pd.get_dummies(df_processed, columns=['Geography'], drop_first=True)
    
    return df_processed, le_gender


def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )
    }
    
    trained_models = {}
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5,
                                scoring='roc_auc', n_jobs=-1)
        cv_results[name] = {'mean_auc': scores.mean(), 'std_auc': scores.std()}
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models, cv_results


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'Accuracy': accuracy, 'Precision': precision,
        'Recall': recall, 'F1': f1, 'ROC AUC': roc_auc,
        'predictions': y_pred, 'probabilities': y_pred_proba
    }


def main():
    st.sidebar.markdown("# Control Panel")
    st.sidebar.markdown("---")
    
    df = load_data()
    
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            Customer Churn Prediction
        </h1>
        <p style='color: rgba(255,255,255,0.8); text-align: center; margin: 10px 0 0 0;'>
            Machine Learning Dashboard for Churn Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "EDA", "Predictions", "Model Comparison", 
        "Feature Importance", "Predict"
    ])
    
    with tab1:
        st.markdown("## Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Churn Rate", f"{df['Exited'].mean()*100:.1f}%")
        with col3:
            st.metric("Avg Credit Score", f"{df['CreditScore'].mean():.0f}")
        with col4:
            st.metric("Avg Age", f"{df['Age'].mean():.1f}")
        
        st.markdown("---")
        st.markdown("### Churn Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, values=df['Exited'].value_counts().values,
                        names=['Stayed', 'Churned'],
                        template='plotly_dark',
                        color_discrete_sequence=['#667eea', '#e74c3c'],
                        hole=0.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            churn_by_geo = df.groupby(['Geography', 'Exited']).size().reset_index(name='Count')
            fig = px.bar(churn_by_geo, x='Geography', y='Count', color='Exited',
                        template='plotly_dark', barmode='group',
                        color_discrete_sequence=['#667eea', '#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Key Metrics by Churn Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_balance_churned = df[df['Exited']==1]['Balance'].mean()
            avg_balance_stayed = df[df['Exited']==0]['Balance'].mean()
            st.metric("Avg Balance (Churned)", f"${avg_balance_churned:,.2f}", 
                      delta=f"${avg_balance_churned - avg_balance_stayed:,.2f} vs Stayed")
        with col2:
            avg_age_churned = df[df['Exited']==1]['Age'].mean()
            avg_age_stayed = df[df['Exited']==0]['Age'].mean()
            st.metric("Avg Age (Churned)", f"{avg_age_churned:.1f}", 
                      delta=f"{avg_age_churned - avg_age_stayed:.1f} vs Stayed")
        with col3:
            avg_salary_churned = df[df['Exited']==1]['EstimatedSalary'].mean()
            avg_salary_stayed = df[df['Exited']==0]['EstimatedSalary'].mean()
            st.metric("Avg Salary (Churned)", f"${avg_salary_churned:,.2f}", 
                      delta=f"${avg_salary_churned - avg_salary_stayed:,.2f} vs Stayed")
    
    with tab2:
        st.markdown("## Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Credit Score Distribution")
            fig = px.histogram(df, x='CreditScore', color='Exited', nbins=30,
                             template='plotly_dark', barmode='overlay',
                             color_discrete_sequence=['#667eea', '#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Age Distribution")
            fig = px.histogram(df, x='Age', color='Exited', nbins=30,
                             template='plotly_dark', barmode='overlay',
                             color_discrete_sequence=['#667eea', '#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Balance Distribution")
            fig = px.histogram(df, x='Balance', color='Exited', nbins=30,
                             template='plotly_dark', barmode='overlay',
                             color_discrete_sequence=['#667eea', '#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Salary Distribution")
            fig = px.histogram(df, x='EstimatedSalary', color='Exited', nbins=30,
                             template='plotly_dark', barmode='overlay',
                             color_discrete_sequence=['#667eea', '#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fig = px.imshow(df[numeric_cols].corr(), 
                       template='plotly_dark', color_continuous_scale='RdBu_r',
                       aspect='auto')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Model Training & Predictions")
        
        df_processed, le_gender = preprocess_data(df)
        
        X = df_processed.drop('Exited', axis=1)
        y = df_processed['Exited']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                            random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        with st.spinner("Training models..."):
            trained_models, cv_results = train_models(X_train_scaled, y_train)
        
        results = {}
        for name, model in trained_models.items():
            results[name] = evaluate_model(model, X_test_scaled, y_test, name)
        
        best_model_name = max(results, key=lambda x: results[x]['ROC AUC'])
        best_model = trained_models[best_model_name]
        
        st.success(f"Best Model: **{best_model_name}** (ROC AUC: {results[best_model_name]['ROC AUC']:.4f})")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{results[best_model_name]['Accuracy']:.4f}")
        with col2:
            st.metric("Test Precision", f"{results[best_model_name]['Precision']:.4f}")
        with col3:
            st.metric("Test Recall", f"{results[best_model_name]['Recall']:.4f}")
        with col4:
            st.metric("Test F1", f"{results[best_model_name]['F1']:.4f}")
        
        st.markdown("### Confusion Matrix")
        
        col1, col2, col3 = st.columns(3)
        for i, name in enumerate(['Logistic Regression', 'Random Forest', 'XGBoost']):
            if name in results:
                with [col1, col2, col3][i]:
                    cm = confusion_matrix(y_test, results[name]['predictions'])
                    fig = px.imshow(cm, template='plotly_dark', color_continuous_scale='Blues',
                                   text_auto=True, labels=dict(x="Predicted", y="Actual"))
                    fig.update_layout(title=name, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### ROC Curves")
        fig = go.Figure()
        for name, res in results.items():
            fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC = {res['ROC AUC']:.4f})",
                                    line=dict(width=2)))
        
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', 
                                line=dict(color='white', dash='dash', width=1)))
        fig.update_layout(title='ROC Curves Comparison', xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate', template='plotly_dark',
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## Model Comparison")
        
        metrics_df = pd.DataFrame({name: {k: v for k, v in res.items() 
                                  if k not in ['predictions', 'probabilities']}
                                  for name, res in results.items()}).T
        
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']),
                    use_container_width=True)
        
        fig = go.Figure()
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']:
            fig.add_trace(go.Bar(name=metric, x=list(results.keys()),
                                y=[results[name][metric] for name in results.keys()]))
        
        fig.update_layout(barmode='group', template='plotly_dark',
                         title='Model Metrics Comparison',
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Cross-Validation Results")
        cv_df = pd.DataFrame(cv_results).T
        cv_df['mean_auc'] = cv_df['mean_auc'].round(4)
        cv_df['std_auc'] = cv_df['std_auc'].round(4)
        st.dataframe(cv_df, use_container_width=True)
    
    with tab5:
        st.markdown("## Feature Importance")
        
        model_to_explain = st.selectbox("Select Model", list(trained_models.keys()))
        
        if hasattr(trained_models[model_to_explain], 'feature_importances_'):
            importances = pd.Series(trained_models[model_to_explain].feature_importances_,
                                   index=X.columns).sort_values(ascending=False)
            
            fig = px.bar(x=importances.head(10).values, y=importances.head(10).index,
                        orientation='h', template='plotly_dark',
                        color=importances.head(10).values,
                        color_continuous_scale='viridis')
            fig.update_layout(title=f'Top 10 Feature Importances - {model_to_explain}',
                             xaxis_title='Importance', yaxis_title='Feature',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             height=500)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Feature Importance Table")
            importance_df = pd.DataFrame({
                'Feature': importances.index,
                'Importance': importances.values
            })
            st.dataframe(importance_df, use_container_width=True)
    
    with tab6:
        st.markdown("## Make Predictions")
        
        st.markdown("### Enter Customer Details")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                credit_score = st.slider("Credit Score", 300, 850, 650)
                age = st.slider("Age", 18, 92, 37)
                tenure = st.slider("Tenure (years)", 0, 10, 5)
            
            with col2:
                balance = st.number_input("Balance ($)", 0.0, 250000.0, 75000.0)
                num_products = st.slider("Number of Products", 1, 4, 1)
                has_cr_card = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            with col3:
                is_active = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 100000.0)
                geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
            
            submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            input_data = pd.DataFrame({
                'CreditScore': [credit_score],
                'Gender': [le_gender.transform([gender])[0]],
                'Age': [age],
                'Tenure': [tenure],
                'Balance': [balance],
                'NumOfProducts': [num_products],
                'HasCrCard': [has_cr_card],
                'IsActiveMember': [is_active],
                'EstimatedSalary': [estimated_salary],
                'Geography_Germany': [1 if geography == 'Germany' else 0],
                'Geography_Spain': [1 if geography == 'Spain' else 0]
            })
            
            input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
            
            prediction = best_model.predict(input_scaled)[0]
            probability = best_model.predict_proba(input_scaled)[0][1]
            
            st.markdown("---")
            st.markdown("### Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"Customer Will Churn (Probability: {probability*100:.1f}%)")
                else:
                    st.success(f"Customer Will Stay (Probability: {(1-probability)*100:.1f}%)")
            
            with col2:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    title={'text': "Churn Probability"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#e74c3c" if probability > 0.5 else "#667eea"},
                        'steps': [
                            {'range': [0, 30], 'color': "#2ecc71"},
                            {'range': [30, 70], 'color': "#f39c12"},
                            {'range': [70, 100], 'color': "#e74c3c"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': probability * 100
                        }
                    }
                ))
                fig.update_layout(height=300, template='plotly_dark',
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download")
    
    if st.sidebar.button("Save Model"):
        joblib.dump(best_model, 'best_churn_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.sidebar.success("Model saved!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Author:** Alireza Sepehri  
    **Email:** alireza_sepehri@mathdep.iust.ac.ir  
    **GitHub:** [@AlirezasDev](https://github.com/AlirezasDev)
    """)


if __name__ == "__main__":
    main()