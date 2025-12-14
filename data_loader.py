import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

@st.cache_data(ttl=3600, show_spinner=True)
def load_data():
    """Load and preprocess main analytics data"""
    try:
        # Try to load the new CSV file first
        df = pd.read_csv('segmented_loan_data.csv')
        st.success("✓ Loaded segmented loan data successfully")
        
        # Ensure proper data types
        numeric_cols = ['AGE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CREDIT_INCOME_RATIO']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create additional features if not present
        if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
            df['DEBT_TO_INCOME'] = (
                df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
            )
        
        # Create risk segments if not present
        if 'CREDIT_INCOME_RATIO' in df.columns and 'RISK_SEGMENT' not in df.columns:
            conditions = [
                df['CREDIT_INCOME_RATIO'] <= 1.5,
                df['CREDIT_INCOME_RATIO'] <= 3,
                df['CREDIT_INCOME_RATIO'] <= 5,
                df['CREDIT_INCOME_RATIO'] > 5
            ]
            choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
            df['RISK_SEGMENT'] = np.select(conditions, choices, default='Medium Risk')
        
        # Create demographic segments if not present
        if all(col in df.columns for col in ['AGE', 'AMT_INCOME_TOTAL']) and 'DEMOGRAPHIC_SEGMENT' not in df.columns:
            df['INCOME_CATEGORY'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=3, labels=['Low', 'Medium', 'High'])
            conditions = [
                (df['AGE'] < 30) & (df['INCOME_CATEGORY'] == 'Low'),
                (df['AGE'] < 30) & (df['INCOME_CATEGORY'].isin(['Medium', 'High'])),
                (df['AGE'].between(30, 50)) & (df['INCOME_CATEGORY'].isin(['Medium', 'High'])),
                (df['AGE'].between(30, 50)) & (df['INCOME_CATEGORY'] == 'Low'),
                (df['AGE'] > 50)
            ]
            choices = [
                'Young & Low Income', 
                'Young & High Income',
                'Middle-Aged & Educated',
                'Middle-Aged & Less Educated', 
                'Senior'
            ]
            df['DEMOGRAPHIC_SEGMENT'] = np.select(conditions, choices, default='Middle-Aged & Educated')
        
        return df
        
    except FileNotFoundError:
        try:
            # Fallback to original data
            # st.warning("Segmented data not found, loading original data...")
            df = pd.read_csv('application_data.csv')
            
            # Apply preprocessing
            if 'DAYS_BIRTH' in df.columns:
                df['AGE'] = (-df['DAYS_BIRTH'] / 365.25).astype(float)
            
            if all(col in df.columns for col in ['AMT_CREDIT', 'AMT_INCOME_TOTAL']):
                df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)
                df['CREDIT_INCOME_RATIO'] = df['CREDIT_INCOME_RATIO'].clip(0, 20)
                
                # Create risk segments
                conditions = [
                    df['CREDIT_INCOME_RATIO'] <= 1.5,
                    df['CREDIT_INCOME_RATIO'] <= 3,
                    df['CREDIT_INCOME_RATIO'] <= 5,
                    df['CREDIT_INCOME_RATIO'] > 5
                ]
                choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
                df['RISK_SEGMENT'] = np.select(conditions, choices, default='Medium Risk')
                
                # Create demographic segments
                df['INCOME_CATEGORY'] = pd.qcut(df['AMT_INCOME_TOTAL'].rank(method='first'), 
                                               q=3, labels=['Low', 'Medium', 'High'])
                conditions = [
                    (df['AGE'] < 30) & (df['INCOME_CATEGORY'] == 'Low'),
                    (df['AGE'] < 30) & (df['INCOME_CATEGORY'].isin(['Medium', 'High'])),
                    (df['AGE'].between(30, 50)) & (df['INCOME_CATEGORY'].isin(['Medium', 'High'])),
                    (df['AGE'].between(30, 50)) & (df['INCOME_CATEGORY'] == 'Low'),
                    (df['AGE'] > 50)
                ]
                choices = [
                    'Young & Low Income', 
                    'Young & High Income',
                    'Middle-Aged & Educated',
                    'Middle-Aged & Less Educated', 
                    'Senior'
                ]
                df['DEMOGRAPHIC_SEGMENT'] = np.select(conditions, choices, default='Middle-Aged & Educated')
            
            return df
            
        except FileNotFoundError:
            st.error("No data files found. Using sample data.")
            return generate_sample_data()

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_samples = 5000
    
    # Generate realistic data
    data = pd.DataFrame({
        'SK_ID_CURR': range(100000, 100000 + n_samples),
        'TARGET': np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),
        'CODE_GENDER': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),
        'AGE': np.random.normal(40, 10, n_samples).clip(20, 70).astype(int),
        'AMT_INCOME_TOTAL': np.random.lognormal(11, 0.5, n_samples).clip(20000, 500000),
        'AMT_CREDIT': np.random.lognormal(12, 0.4, n_samples).clip(50000, 2000000),
        'CREDIT_INCOME_RATIO': np.random.exponential(2, n_samples).clip(0, 20),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], n_samples, p=[0.9, 0.1]),
        'NAME_CASH_LOAN_PURPOSE': np.random.choice([
            'XNA', 'Repairs', 'Education', 'Car', 'Home', 'Business', 'Wedding', 'Other'
        ], n_samples, p=[0.1, 0.15, 0.1, 0.2, 0.15, 0.1, 0.05, 0.15]),
        'OCCUPATION_TYPE': np.random.choice([
            'Laborers', 'Core staff', 'Sales staff', 'Managers', 'Drivers',
            'High skill tech staff', 'Accountants', 'Medicine staff', 'Cooking staff', 'Security staff'
        ], n_samples, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.05])
    })
    
    # Create risk segments
    conditions = [
        data['CREDIT_INCOME_RATIO'] <= 1.5,
        data['CREDIT_INCOME_RATIO'] <= 3,
        data['CREDIT_INCOME_RATIO'] <= 5,
        data['CREDIT_INCOME_RATIO'] > 5
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    data['RISK_SEGMENT'] = np.select(conditions, choices, default='Medium Risk')
    
    # Create demographic segments
    data['INCOME_CATEGORY'] = pd.qcut(data['AMT_INCOME_TOTAL'].rank(method='first'), 
                                     q=3, labels=['Low', 'Medium', 'High'])
    conditions = [
        (data['AGE'] < 30) & (data['INCOME_CATEGORY'] == 'Low'),
        (data['AGE'] < 30) & (data['INCOME_CATEGORY'].isin(['Medium', 'High'])),
        (data['AGE'].between(30, 50)) & (data['INCOME_CATEGORY'].isin(['Medium', 'High'])),
        (data['AGE'].between(30, 50)) & (data['INCOME_CATEGORY'] == 'Low'),
        (data['AGE'] > 50)
    ]
    choices = [
        'Young & Low Income', 
        'Young & High Income',
        'Middle-Aged & Educated',
        'Middle-Aged & Less Educated', 
        'Senior'
    ]
    data['DEMOGRAPHIC_SEGMENT'] = np.select(conditions, choices, default='Middle-Aged & Educated')
    
    st.warning("⚠️ Using generated sample data")
    return data

@st.cache_data
def load_preprocessed_data():
    """Load preprocessed Q&A data"""
    if not os.path.exists('preprocessed_data.pkl'):
        st.error("❌ Preprocessed data not found! Please run `python preprocess.py` first to generate the preprocessed data.")
        raise FileNotFoundError("preprocessed_data.pkl not found")
    
    with open('preprocessed_data.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data_for_overview():
    """Load data for Q&A overview"""
    app_data = pd.read_csv('application_data.csv')
    prev_data = pd.read_csv('previous_application.csv')
    return app_data, prev_data