import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

st.set_page_config(page_title="Loan Risk Analysis Dashboard", layout="wide")

@st.cache_data
def load_preprocessed_data():
    if not os.path.exists('preprocessed_data.pkl'):
        st.error("❌ Preprocessed data not found! Please run `python preprocess.py` first to generate the preprocessed data.")
        st.stop()
    with open('preprocessed_data.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data_for_overview():
    app_data = pd.read_csv('application_data.csv')
    prev_data = pd.read_csv('previous_application.csv')
    return app_data, prev_data

preprocessed = load_preprocessed_data()
app_data, prev_data = load_data_for_overview()

st.title("🏦 Loan Risk Analysis Dashboard")
st.markdown("---")

st.sidebar.header("Navigation")
questions = [
    "Overview",
    "Exploratory Analysis",
    "Q6: Income & Credit Ratios",
    "Q7: Employment Tenure Impact",
    "Q8: Employment Types & Occupations",
    "Q9: Previous Loan Outcomes",
    "Q10: Credit Bureau Enquiry Frequency"
]
selected = st.sidebar.radio("Select Analysis", questions)

if selected == "Overview":
    st.header("📊 Dataset Overview")
    overview = preprocessed['overview']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Applications", f"{overview['total_applications']:,}")
    with col2:
        st.metric("Overall Default Rate", f"{overview['default_rate']:.2f}%")
    with col3:
        st.metric("Previous Applications", f"{overview['total_prev_applications']:,}")
    
    st.markdown("### Data Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Application Data Columns")
        st.write(f"Total columns: {overview['app_data_columns']}")
        st.dataframe(overview['app_data_head'])
    with col2:
        st.subheader("Previous Application Data Columns")
        st.write(f"Total columns: {overview['prev_data_columns']}")
        st.dataframe(overview['prev_data_head'])

elif selected == "Exploratory Analysis":
    st.header("🔍 Detailed Exploratory Analysis")
    st.markdown("### Comprehensive analysis of customer attributes, loan attributes, and previous loan behavior")
    
    st.markdown("#### Customer Attributes Analysis")
    
    customer_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
                     'AMT_INCOME_TOTAL', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                     'NAME_HOUSING_TYPE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'NAME_INCOME_TYPE', 
                     'OCCUPATION_TYPE']
    
    available_customer_cols = [col for col in customer_cols if col in app_data.columns]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Gender Distribution by Default Status")
        if 'CODE_GENDER' in app_data.columns:
            gender_analysis = app_data.groupby('CODE_GENDER')['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            gender_analysis.columns = ['Gender', 'Default Rate (%)', 'Count']
            fig = px.bar(
                gender_analysis,
                x='Gender',
                y='Default Rate (%)',
                title='Default Rate by Gender',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Housing Type Analysis")
        if 'NAME_HOUSING_TYPE' in app_data.columns:
            housing_analysis = app_data.groupby('NAME_HOUSING_TYPE')['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            housing_analysis.columns = ['Housing Type', 'Default Rate (%)', 'Count']
            housing_analysis = housing_analysis.sort_values('Default Rate (%)', ascending=False)
            fig = px.bar(
                housing_analysis,
                x='Housing Type',
                y='Default Rate (%)',
                title='Default Rate by Housing Type',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Education Level Analysis")
    if 'NAME_EDUCATION_TYPE' in app_data.columns:
        education_analysis = app_data.groupby('NAME_EDUCATION_TYPE')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        education_analysis.columns = ['Education Type', 'Default Rate (%)', 'Count']
        education_analysis = education_analysis.sort_values('Default Rate (%)', ascending=False)
        fig = px.bar(
            education_analysis,
            x='Education Type',
            y='Default Rate (%)',
            title='Default Rate by Education Level',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Family Status Analysis")
    if 'NAME_FAMILY_STATUS' in app_data.columns:
        family_analysis = app_data.groupby('NAME_FAMILY_STATUS')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        family_analysis.columns = ['Family Status', 'Default Rate (%)', 'Count']
        family_analysis = family_analysis.sort_values('Default Rate (%)', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                family_analysis,
                x='Family Status',
                y='Default Rate (%)',
                title='Default Rate by Family Status',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(family_analysis.style.format({'Default Rate (%)': '{:.2f}%'}), use_container_width=True)
    
    st.markdown("#### Loan Attributes Analysis")
    
    loan_cols = ['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_CONTRACT_TYPE', 
                 'AMT_DOWN_PAYMENT', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Contract Type Analysis")
        if 'NAME_CONTRACT_TYPE' in app_data.columns:
            contract_analysis = app_data.groupby('NAME_CONTRACT_TYPE')['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            contract_analysis.columns = ['Contract Type', 'Default Rate (%)', 'Count']
            contract_analysis = contract_analysis.sort_values('Default Rate (%)', ascending=False)
            fig = px.bar(
                contract_analysis,
                x='Contract Type',
                y='Default Rate (%)',
                title='Default Rate by Contract Type',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Credit Amount Distribution")
        if 'AMT_CREDIT' in app_data.columns:
            credit_bins = pd.qcut(app_data['AMT_CREDIT'], q=5, duplicates='drop')
            credit_analysis = app_data.groupby(credit_bins)['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count'),
                ('mean_credit', lambda x: app_data.loc[x.index, 'AMT_CREDIT'].mean())
            ]).reset_index()
            credit_analysis.columns = ['Credit Range', 'Default Rate (%)', 'Count', 'Mean Credit']
            credit_analysis['Credit Range'] = credit_analysis['Credit Range'].astype(str)
            fig = px.bar(
                credit_analysis,
                x='Credit Range',
                y='Default Rate (%)',
                title='Default Rate by Credit Amount Quintiles',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Application Timing Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'WEEKDAY_APPR_PROCESS_START' in app_data.columns:
            weekday_analysis = app_data.groupby('WEEKDAY_APPR_PROCESS_START')['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            weekday_analysis.columns = ['Weekday', 'Default Rate (%)', 'Count']
            weekday_order = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
            weekday_analysis['Weekday'] = pd.Categorical(weekday_analysis['Weekday'], categories=weekday_order, ordered=True)
            weekday_analysis = weekday_analysis.sort_values('Weekday')
            fig = px.bar(
                weekday_analysis,
                x='Weekday',
                y='Default Rate (%)',
                title='Default Rate by Application Weekday',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'HOUR_APPR_PROCESS_START' in app_data.columns:
            hour_bins = pd.cut(app_data['HOUR_APPR_PROCESS_START'], bins=[0, 6, 12, 18, 24], labels=['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])
            hour_analysis = app_data.groupby(hour_bins)['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            hour_analysis.columns = ['Hour Range', 'Default Rate (%)', 'Count']
            fig = px.bar(
                hour_analysis,
                x='Hour Range',
                y='Default Rate (%)',
                title='Default Rate by Application Hour',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Previous Loan Behavior Analysis")
    
    prev_count = prev_data.groupby('SK_ID_CURR').size().reset_index(name='prev_loan_count')
    prev_merged = app_data[['SK_ID_CURR', 'TARGET']].merge(prev_count, on='SK_ID_CURR', how='left')
    prev_merged['has_previous'] = prev_merged['prev_loan_count'].notna()
    prev_merged['prev_loan_count'] = prev_merged['prev_loan_count'].fillna(0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Previous Loan Count Impact")
        prev_count_analysis = prev_merged[prev_merged['has_previous']].copy()
        if len(prev_count_analysis) > 0:
            prev_count_bins = pd.cut(prev_count_analysis['prev_loan_count'], bins=[0, 1, 3, 5, 100], labels=['1', '2-3', '4-5', '6+'])
            count_analysis = prev_count_analysis.groupby(prev_count_bins)['TARGET'].agg([
                ('default_rate', lambda x: (x.sum() / len(x)) * 100),
                ('count', 'count')
            ]).reset_index()
            count_analysis.columns = ['Previous Loan Count', 'Default Rate (%)', 'Count']
            count_analysis = count_analysis[count_analysis['Count'] > 0]
            if len(count_analysis) > 0:
                fig = px.bar(
                    count_analysis,
                    x='Previous Loan Count',
                    y='Default Rate (%)',
                    title='Default Rate by Number of Previous Loans',
                    color='Default Rate (%)',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Previous vs No Previous Loans")
        prev_comparison = prev_merged.groupby('has_previous')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        prev_comparison['Category'] = prev_comparison['has_previous'].map({
            False: 'No Previous Loans',
            True: 'Has Previous Loans'
        })
        fig = px.bar(
            prev_comparison,
            x='Category',
            y='default_rate',
            title='Default Rate: Previous Loans vs No History',
            color='Category',
            color_discrete_map={
                'No Previous Loans': '#4ECDC4',
                'Has Previous Loans': '#FF6B6B'
            }
        )
        fig.update_layout(yaxis_title='Default Rate (%)', showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Previous Loan Status Analysis")
    if 'NAME_CONTRACT_STATUS' in prev_data.columns:
        prev_status_first = prev_data.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].first().reset_index()
        prev_status_merged = app_data[['SK_ID_CURR', 'TARGET']].merge(
            prev_status_first,
            on='SK_ID_CURR',
            how='inner'
        )
        status_analysis = prev_status_merged.groupby('NAME_CONTRACT_STATUS')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        status_analysis.columns = ['Previous Status', 'Default Rate (%)', 'Count']
        status_analysis = status_analysis.sort_values('Default Rate (%)', ascending=False)
        if len(status_analysis) > 0:
            fig = px.bar(
                status_analysis,
                x='Previous Status',
                y='Default Rate (%)',
                title='Default Rate by Previous Loan Status',
                color='Default Rate (%)',
                color_continuous_scale='Reds'
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("##### Summary Statistics")
    summary_stats = {
        'Customer Attributes': {
            'Total Customers': len(app_data),
            'Gender Distribution': app_data['CODE_GENDER'].value_counts().to_dict() if 'CODE_GENDER' in app_data.columns else 'N/A',
            'Average Income': f"${app_data['AMT_INCOME_TOTAL'].mean():,.2f}" if 'AMT_INCOME_TOTAL' in app_data.columns else 'N/A',
            'Average Age': f"{(-app_data['DAYS_BIRTH'].mean() / 365.25):.1f} years" if 'DAYS_BIRTH' in app_data.columns else 'N/A'
        },
        'Loan Attributes': {
            'Average Credit Amount': f"${app_data['AMT_CREDIT'].mean():,.2f}" if 'AMT_CREDIT' in app_data.columns else 'N/A',
            'Average Annuity': f"${app_data['AMT_ANNUITY'].mean():,.2f}" if 'AMT_ANNUITY' in app_data.columns else 'N/A',
            'Contract Types': app_data['NAME_CONTRACT_TYPE'].value_counts().to_dict() if 'NAME_CONTRACT_TYPE' in app_data.columns else 'N/A'
        },
        'Previous Loan Behavior': {
            'Customers with Previous Loans': prev_merged['has_previous'].sum() if 'has_previous' in prev_merged.columns else 'N/A',
            'Percentage with Previous Loans': f"{(prev_merged['has_previous'].sum() / len(prev_merged) * 100):.2f}%" if 'has_previous' in prev_merged.columns else 'N/A'
        }
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Customer Attributes**")
        for key, value in summary_stats['Customer Attributes'].items():
            st.write(f"{key}: {value}")
    with col2:
        st.markdown("**Loan Attributes**")
        for key, value in summary_stats['Loan Attributes'].items():
            st.write(f"{key}: {value}")
    with col3:
        st.markdown("**Previous Loan Behavior**")
        for key, value in summary_stats['Previous Loan Behavior'].items():
            st.write(f"{key}: {value}")

elif selected == "Q6: Income & Credit Ratios":
    st.header("Q6: Do higher credit-to-income or annuity-to-income ratios signal stronger repayment capacity?")
    
    q1 = preprocessed['q1']
    credit_income_analysis = q1['credit_income_analysis']
    annuity_income_analysis = q1['annuity_income_analysis']
    corr_credit_income = q1['corr_credit_income']
    corr_annuity_income = q1['corr_annuity_income']
    high_credit_income = q1['high_credit_income']
    low_credit_income = q1['low_credit_income']
    high_annuity_income = q1['high_annuity_income']
    low_annuity_income = q1['low_annuity_income']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            credit_income_analysis,
            x='Quintile',
            y='Default Rate (%)',
            title='Default Rate by Credit-to-Income Ratio Quintiles',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Correlation", f"{corr_credit_income:.3f}", 
                 "Positive = Higher ratio → Higher default")
    
    with col2:
        fig = px.bar(
            annuity_income_analysis,
            x='Quintile',
            y='Default Rate (%)',
            title='Default Rate by Annuity-to-Income Ratio Quintiles',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Correlation", f"{corr_annuity_income:.3f}",
                 "Positive = Higher ratio → Higher default")
    
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Credit-to-Income Ratio")
        st.metric("High Ratio (Q5)", f"{high_credit_income:.2f}%", 
                 f"{high_credit_income - low_credit_income:.2f} pp vs Q1")
        if corr_credit_income > 0.1:
            st.success("✅ **Strong signal!** Higher credit-to-income ratios indicate higher default risk.")
        elif corr_credit_income > 0:
            st.info("ℹ️ **Moderate signal:** Some evidence that higher ratios indicate higher risk.")
        else:
            st.warning("⚠️ **Weak signal:** No clear relationship detected.")
    
    with col2:
        st.markdown("#### Annuity-to-Income Ratio")
        st.metric("High Ratio (Q5)", f"{high_annuity_income:.2f}%",
                 f"{high_annuity_income - low_annuity_income:.2f} pp vs Q1")
        if corr_annuity_income > 0.1:
            st.success("✅ **Strong signal!** Higher annuity-to-income ratios indicate higher default risk.")
        elif corr_annuity_income > 0:
            st.info("ℹ️ **Moderate signal:** Some evidence that higher ratios indicate higher risk.")
        else:
            st.warning("⚠️ **Weak signal:** No clear relationship detected.")
    
    st.markdown("### Detailed Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Credit-to-Income Ratio Analysis")
        st.dataframe(credit_income_analysis.style.format({'Default Rate (%)': '{:.2f}%', 'Mean Ratio': '{:.4f}'}), use_container_width=True)
    with col2:
        st.markdown("#### Annuity-to-Income Ratio Analysis")
        st.dataframe(annuity_income_analysis.style.format({'Default Rate (%)': '{:.2f}%', 'Mean Ratio': '{:.4f}'}), use_container_width=True)

elif selected == "Q7: Employment Tenure Impact":
    st.header("Q7: Does employment tenure affect default probability?")
    
    q2 = preprocessed['q2']
    tenure_category_analysis = q2['tenure_category_analysis']
    tenure_quintile_analysis = q2['tenure_quintile_analysis']
    tenure_bin_analysis = q2['tenure_bin_analysis']
    corr_tenure = q2['corr_tenure']
    shortest_tenure = q2['shortest_tenure']
    longest_tenure = q2['longest_tenure']
    diff_tenure = q2['diff_tenure']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            tenure_category_analysis,
            x='Tenure Category',
            y='Default Rate (%)',
            title='Default Rate by Employment Tenure Category',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            tenure_quintile_analysis,
            x='Quintile',
            y='Default Rate (%)',
            title='Default Rate by Employment Tenure Quintiles',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Correlation", f"{corr_tenure:.3f}",
                 "Negative = Longer tenure → Lower default")
    
    st.markdown("### Trend Analysis")
    
    fig = px.scatter(
        tenure_bin_analysis,
        x='Mean Years',
        y='Default Rate (%)',
        size='Count',
        title='Default Rate vs Employment Tenure (Years)',
        trendline='ols',
        hover_data=['Count']
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Key Insights")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shortest Tenure (< 6 months)", f"{shortest_tenure:.2f}%", "Highest risk")
    with col2:
        st.metric("Longest Tenure (10+ years)", f"{longest_tenure:.2f}%", "Lowest risk")
    with col3:
        st.metric("Risk Difference", f"{diff_tenure:.2f} pp", 
                 "Impact of tenure")
    
    if corr_tenure < -0.1:
        st.success("✅ **Strong effect!** Longer employment tenure significantly reduces default probability.")
    elif corr_tenure < 0:
        st.info("ℹ️ **Moderate effect:** Some evidence that longer tenure reduces default risk.")
    else:
        st.warning("⚠️ **Weak effect:** Limited relationship between tenure and default probability.")
    
    st.markdown("### Detailed Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### By Category")
        st.dataframe(tenure_category_analysis.style.format({'Default Rate (%)': '{:.2f}%', 'Mean Years': '{:.2f}'}), use_container_width=True)
    with col2:
        st.markdown("#### By Quintile")
        st.dataframe(tenure_quintile_analysis.style.format({'Default Rate (%)': '{:.2f}%', 'Mean Years': '{:.2f}'}), use_container_width=True)

elif selected == "Q8: Employment Types & Occupations":
    st.header("Q8: Are specific employment types or occupations higher risk?")
    
    q3 = preprocessed['q3']
    income_type_analysis = q3['income_type_analysis']
    occupation_analysis = q3['occupation_analysis']
    top_risk_income = q3['top_risk_income']
    top_risk_occ = q3['top_risk_occ']
    highest_income = q3['highest_income']
    lowest_income = q3['lowest_income']
    highest_occ = q3['highest_occ']
    lowest_occ = q3['lowest_occ']
    risk_diff_income = q3['risk_diff_income']
    risk_diff_occ = q3['risk_diff_occ']
    risk_range_income = q3['risk_range_income']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            income_type_analysis,
            x='Income Type',
            y='Default Rate (%)',
            title='Default Rate by Income Type',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top Risk Income Types")
        for idx, row in top_risk_income.iterrows():
            st.write(f"**{row['Income Type']}**: {row['Default Rate (%)']:.2f}% (n={row['Count']:,})")
    
    with col2:
        fig = px.bar(
            occupation_analysis,
            x='Occupation Type',
            y='Default Rate (%)',
            title='Default Rate by Occupation Type',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Top Risk Occupations")
        for idx, row in top_risk_occ.iterrows():
            st.write(f"**{row['Occupation Type']}**: {row['Default Rate (%)']:.2f}% (n={row['Count']:,})")
    
    st.markdown("### Risk Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Highest vs Lowest Risk Income Types")
        if highest_income is not None and lowest_income is not None:
            st.metric("Highest Risk", f"{highest_income['Income Type']}", 
                     f"{highest_income['Default Rate (%)']:.2f}%")
            st.metric("Lowest Risk", f"{lowest_income['Income Type']}",
                     f"{lowest_income['Default Rate (%)']:.2f}%")
            
            st.metric("Risk Range", f"{risk_diff_income:.2f} percentage points")
    
    with col2:
        st.markdown("#### Highest vs Lowest Risk Occupations")
        if highest_occ is not None and lowest_occ is not None:
            st.metric("Highest Risk", f"{highest_occ['Occupation Type']}",
                     f"{highest_occ['Default Rate (%)']:.2f}%")
            st.metric("Lowest Risk", f"{lowest_occ['Occupation Type']}",
                     f"{lowest_occ['Default Rate (%)']:.2f}%")
            
            st.metric("Risk Range", f"{risk_diff_occ:.2f} percentage points")
    
    if risk_range_income is not None:
        if risk_range_income > 5:
            st.success("✅ **Significant differences exist in income types!** Employment type is a strong risk indicator.")
        elif risk_range_income > 2:
            st.info("ℹ️ **Moderate differences:** Income types show some variation in risk profiles.")
        else:
            st.warning("⚠️ **Limited differentiation:** Income types show minimal risk differences.")
    
    st.markdown("### Detailed Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Income Types")
        st.dataframe(income_type_analysis.style.format({'Default Rate (%)': '{:.2f}%'}), use_container_width=True)
    with col2:
        st.markdown("#### Occupation Types")
        st.dataframe(occupation_analysis.style.format({'Default Rate (%)': '{:.2f}%'}), use_container_width=True)

elif selected == "Q9: Previous Loan Outcomes":
    st.header("Q9: How do previous loan outcomes (approved, refused, canceled) relate to current default?")
    
    q4 = preprocessed['q4']
    status_analysis = q4['status_analysis']
    approved_analysis = q4['approved_analysis']
    outcome_analysis = q4['outcome_analysis']
    no_prev_rate = q4['no_prev_rate']
    has_prev_rate = q4['has_prev_rate']
    never_approved_rate = q4['never_approved_rate']
    prev_approved_rate = q4['prev_approved_rate']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            status_analysis,
            x='Category',
            y='default_rate',
            title='Default Rate: Previous Applications vs No History',
            color='Category',
            color_discrete_map={
                'No Previous Applications': '#4ECDC4',
                'Has Previous Applications': '#FF6B6B'
            },
            text='default_rate'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400, yaxis_title='Default Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            approved_analysis,
            x='Category',
            y='default_rate',
            title='Default Rate: Previous Approvals Impact',
            color='Category',
            color_discrete_map={
                'Never Approved': '#FF6B6B',
                'Previously Approved': '#4ECDC4'
            },
            text='default_rate'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(showlegend=False, height=400, yaxis_title='Default Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Detailed Outcome Analysis")
    
    fig = px.bar(
        outcome_analysis,
        x='Previous Outcome Pattern',
        y='Default Rate (%)',
        title='Default Rate by Previous Loan Outcome Patterns',
        color='Default Rate (%)',
        color_continuous_scale='Reds',
        text='Default Rate (%)'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Previous Application History")
        st.metric("No Previous Apps", f"{no_prev_rate:.2f}%", 
                 f"{has_prev_rate - no_prev_rate:.2f} pp difference")
        st.metric("Has Previous Apps", f"{has_prev_rate:.2f}%", 
                 "vs no history")
        if has_prev_rate > no_prev_rate + 2:
            st.warning("⚠️ **Previous applications increase risk**")
        elif has_prev_rate < no_prev_rate - 2:
            st.success("✅ **Previous applications decrease risk**")
        else:
            st.info("ℹ️ **Limited impact** of previous application history")
    
    with col2:
        st.markdown("#### Approval Status Impact")
        st.metric("Never Approved", f"{never_approved_rate:.2f}%",
                 f"{never_approved_rate - prev_approved_rate:.2f} pp vs approved")
        st.metric("Previously Approved", f"{prev_approved_rate:.2f}%",
                 "Lower risk indicator")
        if prev_approved_rate < never_approved_rate - 2:
            st.success("✅ **Previous approval is a positive signal!**")
        else:
            st.info("ℹ️ **Limited signal** from approval status")
    
    st.markdown("### Detailed Statistics")
    st.dataframe(outcome_analysis.style.format({'Default Rate (%)': '{:.2f}%'}), use_container_width=True)

elif selected == "Q10: Credit Bureau Enquiry Frequency":
    st.header("Q10: Does having credit bureau enquiries correlate with higher risk?")
    
    q5 = preprocessed['q5']
    total_enquiry_analysis = q5.get('total_enquiry_analysis', pd.DataFrame())
    corr_total = q5.get('corr_total', np.nan)
    zero_enq_rate = q5.get('zero_enq_rate', None)
    non_zero_enq_rate = q5.get('non_zero_enq_rate', None)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            total_enquiry_analysis,
            x='Total Enquiries',
            y='Default Rate (%)',
            title='Default Rate: 0 Enquiries vs Non-zero Enquiries',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        if not np.isnan(corr_total):
            if corr_total > 0:
                hint = "↑ Positive = Having enquiries → Higher risk"
            elif corr_total < 0:
                hint = "↓ Negative = Having enquiries → Lower risk"
            else:
                hint = "No correlation detected"
            st.metric("Correlation", f"{corr_total:.3f}", hint)
        else:
            st.metric("Correlation", "N/A", "Insufficient data variation")
    
    with col2:
        st.markdown("### Comparison")
        
        if zero_enq_rate is not None:
            st.metric("0 Enquiries", f"{zero_enq_rate:.2f}%", "Default rate")
        else:
            st.metric("0 Enquiries", "N/A", "No data")
        
        if non_zero_enq_rate is not None:
            diff = non_zero_enq_rate - zero_enq_rate if zero_enq_rate else None
            st.metric("Non-zero Enquiries", f"{non_zero_enq_rate:.2f}%",
                     f"{diff:+.2f} pp" if diff is not None else "")
        else:
            st.metric("Non-zero Enquiries", "N/A", "No data")
        
        if zero_enq_rate is not None and non_zero_enq_rate is not None:
            diff_pp = non_zero_enq_rate - zero_enq_rate
            if diff_pp > 2:
                st.success(f"✅ **Higher risk with enquiries:** Non-zero enquiries have {diff_pp:.2f} percentage points higher default rate.")
            elif diff_pp < -2:
                st.info(f"ℹ️ **Lower risk with enquiries:** Non-zero enquiries have {abs(diff_pp):.2f} percentage points lower default rate.")
            else:
                st.info(f"ℹ️ **Similar risk:** Difference is only {abs(diff_pp):.2f} percentage points.")
    
    st.markdown("### Key Insights")
    
    if not np.isnan(corr_total):
        if corr_total > 0.1:
            st.success("✅ **Strong positive correlation!** Having credit bureau enquiries is associated with higher default risk.")
        elif corr_total > 0.05:
            st.info("ℹ️ **Moderate positive correlation:** Some evidence that having enquiries correlates with higher risk.")
        elif corr_total < -0.1:
            st.warning("⚠️ **Strong negative correlation:** Having enquiries correlates with lower risk. This may indicate that applicants with enquiry history have better credit management.")
        elif corr_total < -0.05:
            st.info("ℹ️ **Moderate negative correlation:** Some evidence that having enquiries correlates with lower risk.")
        else:
            st.info("ℹ️ **Weak correlation:** Limited relationship between having enquiries and default risk.")
    else:
        st.warning("⚠️ **Unable to calculate correlation:** Insufficient data variation.")
    
    st.markdown("### Detailed Statistics")
    st.dataframe(total_enquiry_analysis.style.format({'Default Rate (%)': '{:.2f}%'}), use_container_width=True)

st.markdown("---")
st.markdown("### 📝 Notes")
st.info("This dashboard analyzes loan default risk based on various factors. Use the sidebar to navigate between different analyses.")
