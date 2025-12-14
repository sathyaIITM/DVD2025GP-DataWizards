import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def create_risk_segment_analysis(df):
    """Create risk segment analysis chart"""
    if 'RISK_SEGMENT' not in df.columns:
        return go.Figure()
    
    segment_stats = df.groupby('RISK_SEGMENT').agg(
        total=('TARGET', 'count'),
        defaults=('TARGET', 'sum')
    ).reset_index()
    
    segment_stats['default_rate'] = segment_stats['defaults'] / segment_stats['total']
    segment_stats = segment_stats.sort_values('default_rate', ascending=False)
    
    # Color mapping
    color_map = {
        'Low Risk': '#10B981',
        'Medium Risk': '#F59E0B',
        'High Risk': '#EF4444',
        'Very High Risk': '#7C3AED'
    }
    
    colors = [color_map.get(x, '#6B7280') for x in segment_stats['RISK_SEGMENT']]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Default Rate by Risk Segment', 'Application Distribution'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]]
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=segment_stats['RISK_SEGMENT'],
            y=segment_stats['default_rate'],
            name='Default Rate',
            marker_color=colors,
            text=[f'{x:.1%}' for x in segment_stats['default_rate']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(
            labels=segment_stats['RISK_SEGMENT'],
            values=segment_stats['total'],
            name='Distribution',
            hole=0.4,
            marker=dict(colors=colors)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Default Rate", row=1, col=1)
    
    return fig

@st.cache_resource(show_spinner=False)
def create_demographic_analysis(df):
    """Create demographic segment analysis"""
    if 'DEMOGRAPHIC_SEGMENT' not in df.columns:
        return go.Figure()
    
    segment_stats = df.groupby('DEMOGRAPHIC_SEGMENT').agg(
        total=('TARGET', 'count'),
        defaults=('TARGET', 'sum')
    ).reset_index()
    
    segment_stats['default_rate'] = segment_stats['defaults'] / segment_stats['total']
    segment_stats = segment_stats.sort_values('default_rate', ascending=True)
    
    fig = px.bar(
        segment_stats,
        x='default_rate',
        y='DEMOGRAPHIC_SEGMENT',
        orientation='h',
        color='default_rate',
        color_continuous_scale='RdYlGn_r',
        title='Default Rates by Demographic Segment',
        labels={'default_rate': 'Default Rate', 'DEMOGRAPHIC_SEGMENT': 'Segment'},
        hover_data=['total', 'defaults']
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        template='plotly_white'
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_financial_analysis(df):
    """Create financial ratio analysis"""
    if 'CREDIT_INCOME_RATIO' not in df.columns:
        return go.Figure()
    
    # Create bins for credit income ratio
    df_copy = df.copy()
    df_copy['CREDIT_INCOME_BIN'] = pd.cut(
        df_copy['CREDIT_INCOME_RATIO'],
        bins=[0, 1, 2, 3, 5, 10, np.inf],
        labels=['0-1', '1-2', '2-3', '3-5', '5-10', '10+']
    )
    
    ratio_stats = df_copy.groupby('CREDIT_INCOME_BIN').agg(
        total=('TARGET', 'count'),
        defaults=('TARGET', 'sum')
    ).reset_index()
    
    ratio_stats['default_rate'] = ratio_stats['defaults'] / ratio_stats['total']
    
    fig = px.line(
        ratio_stats,
        x='CREDIT_INCOME_BIN',
        y='default_rate',
        markers=True,
        title='Default Rate by Credit-to-Income Ratio',
        labels={'CREDIT_INCOME_BIN': 'Credit/Income Ratio', 'default_rate': 'Default Rate'}
    )
    
    fig.update_traces(
        line=dict(color='#EF4444', width=3),
        marker=dict(size=10, color='#EF4444')
    )
    
    # Add bar for count
    fig.add_trace(
        go.Bar(
            x=ratio_stats['CREDIT_INCOME_BIN'],
            y=ratio_stats['total'],
            name='Applications',
            yaxis='y2',
            marker_color='#3B82F6',
            opacity=0.3
        )
    )
    
    fig.update_layout(
        height=400,
        yaxis=dict(title='Default Rate'),
        yaxis2=dict(
            title='Applications',
            overlaying='y',
            side='right'
        ),
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_age_analysis(df):
    """Create age group analysis"""
    if 'AGE' not in df.columns:
        return go.Figure()
    
    # Create age groups
    df_copy = df.copy()
    df_copy['AGE_GROUP'] = pd.cut(
        df_copy['AGE'],
        bins=[20, 30, 40, 50, 60, 70],
        labels=['20-29', '30-39', '40-49', '50-59', '60+']
    )
    
    age_stats = df_copy.groupby('AGE_GROUP').agg(
        total=('TARGET', 'count'),
        defaults=('TARGET', 'sum'),
        avg_income=('AMT_INCOME_TOTAL', 'mean')
    ).reset_index()
    
    age_stats['default_rate'] = age_stats['defaults'] / age_stats['total']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Default Rate by Age Group', 'Average Income by Age Group'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Default rate
    fig.add_trace(
        go.Bar(
            x=age_stats['AGE_GROUP'],
            y=age_stats['default_rate'],
            name='Default Rate',
            marker_color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#8B5CF6'],
            text=[f'{x:.1%}' for x in age_stats['default_rate']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Average income
    fig.add_trace(
        go.Bar(
            x=age_stats['AGE_GROUP'],
            y=age_stats['avg_income'],
            name='Avg Income',
            marker_color=['#EF4444', '#F59E0B', '#3B82F6', '#10B981', '#8B5CF6'],
            text=[f'${x:,.0f}' for x in age_stats['avg_income']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text="Default Rate", row=1, col=1)
    fig.update_yaxes(title_text="Average Income ($)", row=1, col=2)
    
    return fig

@st.cache_resource(show_spinner=False)
def create_income_vs_credit_scatter(df):
    """Create income vs credit scatter plot (optimized for no WebGL)"""
    if 'AMT_INCOME_TOTAL' not in df.columns or 'AMT_CREDIT' not in df.columns:
        return go.Figure()
    
    # Sample for performance
    sample_size = min(1000, len(df))
    sample_df = df.sample(sample_size, random_state=42)
    
    # Separate groups for better SVG rendering
    non_default = sample_df[sample_df['TARGET'] == 0]
    default = sample_df[sample_df['TARGET'] == 1]
    
    fig = go.Figure()
    
    # Add non-default points (SVG mode)
    fig.add_trace(go.Scatter(
        x=non_default['AMT_INCOME_TOTAL'],
        y=non_default['AMT_CREDIT'],
        mode='markers',
        name='Paid',
        marker=dict(
            color='#10B981',
            size=6,
            opacity=0.6,
            symbol='circle'
        ),
        hovertemplate='Income: $%{x:,.0f}<br>Credit: $%{y:,.0f}<br>Status: Paid<extra></extra>'
    ))
    
    # Add default points (SVG mode)
    if len(default) > 0:
        fig.add_trace(go.Scatter(
            x=default['AMT_INCOME_TOTAL'],
            y=default['AMT_CREDIT'],
            mode='markers',
            name='Default',
            marker=dict(
                color='#EF4444',
                size=8,
                opacity=0.8,
                symbol='x'
            ),
            hovertemplate='Income: $%{x:,.0f}<br>Credit: $%{y:,.0f}<br>Status: Default<extra></extra>'
        ))
    
    # Add diagonal line for reference (credit = 5x income)
    max_val = max(sample_df['AMT_INCOME_TOTAL'].max(), sample_df['AMT_CREDIT'].max() / 5)
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, 5 * max_val],
            mode='lines',
            name='5x Income Line',
            line=dict(color='gray', dash='dash'),
            opacity=0.5
        )
    )
    
    fig.update_layout(
        title='Income vs Credit Amount',
        xaxis_title='Income ($)',
        yaxis_title='Credit Amount ($)',
        height=500,
        template='plotly_white',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_occupation_analysis(df):
    """Create occupation analysis chart"""
    if 'OCCUPATION_TYPE' not in df.columns:
        return go.Figure()
    
    # Get top 15 occupations by count
    occ_stats = df.groupby('OCCUPATION_TYPE').agg(
        total=('TARGET', 'count'),
        defaults=('TARGET', 'sum')
    ).reset_index()
    
    occ_stats['default_rate'] = occ_stats['defaults'] / occ_stats['total']
    occ_stats = occ_stats.sort_values('total', ascending=False).head(15)
    occ_stats = occ_stats.sort_values('default_rate', ascending=True)
    
    fig = px.bar(
        occ_stats,
        x='default_rate',
        y='OCCUPATION_TYPE',
        orientation='h',
        color='default_rate',
        color_continuous_scale='RdYlGn_r',
        title='Default Rate by Occupation (Top 15 by Volume)',
        labels={'default_rate': 'Default Rate', 'OCCUPATION_TYPE': 'Occupation'},
        hover_data=['total', 'defaults']
    )
    
    fig.update_layout(
        height=500,
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False,
        template='plotly_white'
    )
    
    # Add value labels
    for i, row in occ_stats.iterrows():
        fig.add_annotation(
            x=row['default_rate'],
            y=row['OCCUPATION_TYPE'],
            text=f"{row['default_rate']:.1%}",
            showarrow=False,
            xshift=30,
            font=dict(size=10)
        )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_loan_type_analysis(df):
    """Create loan type analysis (replaces radar chart)"""
    if 'NAME_CONTRACT_TYPE' not in df.columns:
        return go.Figure()
    
    loan_stats = df.groupby('NAME_CONTRACT_TYPE').agg({
        'TARGET': ['count', 'mean', 'sum'],
        'AMT_CREDIT': 'mean',
        'AMT_INCOME_TOTAL': 'mean',
        'CREDIT_INCOME_RATIO': 'mean'
    }).round(3)
    
    loan_stats.columns = ['Count', 'Default_Rate', 'Defaults', 'Avg_Credit', 'Avg_Income', 'Avg_Credit_Income_Ratio']
    loan_stats = loan_stats.reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Default Rate by Loan Type', 'Average Credit Amount',
                       'Average Income', 'Credit-to-Income Ratio'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Subplot 1: Default Rate
    fig.add_trace(
        go.Bar(
            x=loan_stats['NAME_CONTRACT_TYPE'],
            y=loan_stats['Default_Rate'],
            name='Default Rate',
            marker_color=['#EF4444', '#3B82F6', '#10B981', '#F59E0B'][:len(loan_stats)],
            text=[f'{x:.1%}' for x in loan_stats['Default_Rate']],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Subplot 2: Avg Credit
    fig.add_trace(
        go.Bar(
            x=loan_stats['NAME_CONTRACT_TYPE'],
            y=loan_stats['Avg_Credit'],
            name='Avg Credit',
            marker_color=['#EF4444', '#3B82F6', '#10B981', '#F59E0B'][:len(loan_stats)],
            text=[f'${x:,.0f}' for x in loan_stats['Avg_Credit']],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Subplot 3: Avg Income
    fig.add_trace(
        go.Bar(
            x=loan_stats['NAME_CONTRACT_TYPE'],
            y=loan_stats['Avg_Income'],
            name='Avg Income',
            marker_color=['#EF4444', '#3B82F6', '#10B981', '#F59E0B'][:len(loan_stats)],
            text=[f'${x:,.0f}' for x in loan_stats['Avg_Income']],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # Subplot 4: Credit Income Ratio
    fig.add_trace(
        go.Bar(
            x=loan_stats['NAME_CONTRACT_TYPE'],
            y=loan_stats['Avg_Credit_Income_Ratio'],
            name='Credit/Income Ratio',
            marker_color=['#EF4444', '#3B82F6', '#10B981', '#F59E0B'][:len(loan_stats)],
            text=[f'{x:.2f}' for x in loan_stats['Avg_Credit_Income_Ratio']],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Default Rate", row=1, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=2)
    fig.update_yaxes(title_text="Amount ($)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=2)
    
    return fig

@st.cache_resource(show_spinner=False)
def create_purpose_risk_analysis(df):
    """Create loan purpose risk analysis"""
    if 'NAME_CASH_LOAN_PURPOSE' not in df.columns:
        return go.Figure()
    
    purpose_stats = df.groupby('NAME_CASH_LOAN_PURPOSE').agg({
        'TARGET': ['count', 'mean', 'sum'],
        'AMT_CREDIT': 'mean',
        'AMT_INCOME_TOTAL': 'mean'
    }).round(3)
    
    purpose_stats.columns = ['Count', 'Default_Rate', 'Defaults', 'Avg_Credit', 'Avg_Income']
    purpose_stats = purpose_stats.reset_index()
    
    # Filter for significant volumes
    purpose_stats = purpose_stats[purpose_stats['Count'] > df.shape[0] * 0.01]
    purpose_stats = purpose_stats.sort_values('Default_Rate', ascending=False).head(10)
    
    fig = px.bar(
        purpose_stats,
        x='NAME_CASH_LOAN_PURPOSE',
        y='Default_Rate',
        color='Default_Rate',
        color_continuous_scale='RdYlGn_r',
        title='Default Rate by Loan Purpose (Top 10)',
        labels={'NAME_CASH_LOAN_PURPOSE': 'Loan Purpose', 'Default_Rate': 'Default Rate'},
        hover_data=['Count', 'Avg_Credit', 'Avg_Income']
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=45,
        coloraxis_showscale=False,
        template='plotly_white'
    )
    
    # Add value labels
    for i, row in purpose_stats.iterrows():
        fig.add_annotation(
            x=row['NAME_CASH_LOAN_PURPOSE'],
            y=row['Default_Rate'],
            text=f"{row['Default_Rate']:.1%}",
            showarrow=False,
            yshift=10,
            font=dict(size=10)
        )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_correlation_heatmap(df):
    """Create correlation heatmap for key variables"""
    numeric_cols = ['AGE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CREDIT_INCOME_RATIO', 'TARGET']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(numeric_cols) < 2:
        return go.Figure()
    
    corr_matrix = df[numeric_cols].corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlBu_r',
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        hovertemplate='Correlation between %{y} and %{x}: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Key Variables',
        height=500,
        template='plotly_white'
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_default_distribution_by_feature(df, feature):
    """Create distribution plot showing defaults vs non-defaults"""
    if feature not in df.columns or 'TARGET' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    # Distribution for non-defaults
    non_default = df[df['TARGET'] == 0][feature].dropna()
    if len(non_default) > 0:
        fig.add_trace(go.Histogram(
            x=non_default,
            name='Paid',
            opacity=0.7,
            marker_color='#10B981',
            nbinsx=30
        ))
    
    # Distribution for defaults
    default = df[df['TARGET'] == 1][feature].dropna()
    if len(default) > 0:
        fig.add_trace(go.Histogram(
            x=default,
            name='Default',
            opacity=0.7,
            marker_color='#EF4444',
            nbinsx=30
        ))
    
    fig.update_layout(
        title=f'Distribution of {feature} by Default Status',
        xaxis_title=feature,
        yaxis_title='Count',
        barmode='overlay',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

@st.cache_resource(show_spinner=False)
def create_age_credit_interaction(df):
    """
    Age x Credit burden interaction heatmap
    """
    required_cols = [
        'AGE',
        'CREDIT_INCOME_RATIO',
        'TARGET'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return go.Figure()

    df_copy = df.copy()

    df_copy['AGE_BIN'] = pd.cut(
        df_copy['AGE'],
        bins=[20, 30, 40, 50, 60, 70],
        labels=['20-29', '30-39', '40-49', '50-59', '60+']
    )

    df_copy['CIR_BIN'] = pd.qcut(
        df_copy['CREDIT_INCOME_RATIO'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    pivot = df_copy.pivot_table(
        values='TARGET',
        index='AGE_BIN',
        columns='CIR_BIN',
        aggfunc='mean'
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r',
            text=pivot.values,
            texttemplate='%{text:.2f}',
            hovertemplate='Age: %{y}<br>Credit Stress: %{x}<br>Default Rate: %{z:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        title='Default Rate Heatmap: Age vs Credit Stress',
        height=500,
        template='plotly_white'
    )

    return fig

@st.cache_resource(show_spinner=False)
def create_debt_stress_analysis(df):
    """
    Debt Stress Index vs default probability
    """
    required_cols = [
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_INCOME_TOTAL',
        'TARGET'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return go.Figure()

    df_copy = df.copy()

    df_copy['DEBT_STRESS_INDEX'] = (
        df_copy['AMT_ANNUITY'] / df_copy['AMT_INCOME_TOTAL']
    )

    df_copy['STRESS_BIN'] = pd.qcut(
        df_copy['DEBT_STRESS_INDEX'],
        q=5,
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    stats = df_copy.groupby('STRESS_BIN').agg(
        default_rate=('TARGET', 'mean'),
        total=('TARGET', 'count')
    ).reset_index()

    fig = px.bar(
        stats,
        x='STRESS_BIN',
        y='default_rate',
        color='default_rate',
        color_continuous_scale='Reds',
        title='Default Rate by Debt Stress Index',
        text=stats['default_rate'].apply(lambda x: f"{x:.1%}")
    )

    fig.update_layout(
        height=450,
        coloraxis_showscale=False,
        template='plotly_white'
    )

    return fig

@st.cache_resource(show_spinner=False)
def create_education_employment_analysis(df):
    """
    Improved: Education + empirically-derived employment stability
    vs credit/income & default rate
    """
    required_cols = [
        'NAME_EDUCATION_TYPE',
        'NAME_INCOME_TYPE',
        'CREDIT_INCOME_RATIO',
        'AMT_INCOME_TOTAL',
        'TARGET'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return go.Figure()

    df_copy = df.copy()

  
    education_map = {
        'Lower secondary': 1,
        'Secondary / secondary special': 2,
        'Incomplete higher': 3,
        'Higher education': 4,
        'Academic degree': 5
    }

    df_copy['EDU_SCORE'] = (
        df_copy['NAME_EDUCATION_TYPE']
        .map(education_map)
        .fillna(1)
    )

    emp_stats = df_copy.groupby('NAME_INCOME_TYPE').agg(
        total=('TARGET', 'count'),
        default_rate=('TARGET', 'mean'),
        income_std=('AMT_INCOME_TOTAL', 'std')
    ).reset_index()

    # Keep only meaningful groups
    emp_stats = emp_stats[emp_stats['total'] > 1000]

    # Reference medians
    median_default = emp_stats['default_rate'].median()
    median_income_std = emp_stats['income_std'].median()

    # Stability scoring (0–2)
    emp_stats['EMP_STABILITY_SCORE'] = 0
    emp_stats.loc[
        emp_stats['default_rate'] <= median_default,
        'EMP_STABILITY_SCORE'
    ] += 1

    emp_stats.loc[
        emp_stats['income_std'] <= median_income_std,
        'EMP_STABILITY_SCORE'
    ] += 1

    # Map back to main dataframe
    stability_map = dict(
        zip(emp_stats['NAME_INCOME_TYPE'], emp_stats['EMP_STABILITY_SCORE'])
    )

    df_copy['EMP_STABILITY_SCORE'] = (
        df_copy['NAME_INCOME_TYPE']
        .map(stability_map)
        .fillna(0)
    )

    
    df_copy['SOCIO_ECON_SCORE'] = (
        df_copy['EDU_SCORE'] + df_copy['EMP_STABILITY_SCORE']
    )

   
    stats = df_copy.groupby('SOCIO_ECON_SCORE').agg(
        avg_credit_income=('CREDIT_INCOME_RATIO', 'mean'),
        default_rate=('TARGET', 'mean'),
        total=('TARGET', 'count')
    ).reset_index()

    fig = px.scatter(
        stats,
        x='avg_credit_income',
        y='default_rate',
        size='total',
        color='SOCIO_ECON_SCORE',
        color_continuous_scale='Viridis',
        title='Socio-Economic Stability vs Credit Risk',
        labels={
            'avg_credit_income': 'Avg Credit / Income Ratio',
            'default_rate': 'Default Rate',
            'SOCIO_ECON_SCORE': 'Socio-Economic Stability Score'
        }
    )

    fig.update_layout(
        height=520,
        template='plotly_white'
    )

    return fig


@st.cache_resource(show_spinner=False)
def create_living_standard_analysis(df):
    """
    Living standards (housing + car ownership) vs credit/income & default rate
    """
    required_cols = [
        'FLAG_OWN_REALTY',
        'FLAG_OWN_CAR',
        'CREDIT_INCOME_RATIO',
        'TARGET'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return go.Figure()

    df_copy = df.copy()

    # Feature engineering: Living Standard Score
    df_copy['LIVING_STANDARD_SCORE'] = (
        (df_copy['FLAG_OWN_REALTY'] == 'Y').astype(int) +
        (df_copy['FLAG_OWN_CAR'] == 'Y').astype(int)
    )

    df_copy['LIVING_STANDARD_LABEL'] = df_copy['LIVING_STANDARD_SCORE'].map({
        0: 'No House & No Car',
        1: 'Either House or Car',
        2: 'House & Car'
    })

    stats = df_copy.groupby('LIVING_STANDARD_LABEL').agg(
        avg_credit_income=('CREDIT_INCOME_RATIO', 'mean'),
        default_rate=('TARGET', 'mean'),
        total=('TARGET', 'count')
    ).reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Default Rate by Living Standard',
            'Avg Credit/Income Ratio by Living Standard'
        )
    )

    fig.add_trace(
        go.Bar(
            x=stats['LIVING_STANDARD_LABEL'],
            y=stats['default_rate'],
            text=[f"{x:.1%}" for x in stats['default_rate']],
            textposition='auto',
            marker_color='#EF4444'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=stats['LIVING_STANDARD_LABEL'],
            y=stats['avg_credit_income'],
            text=[f"{x:.2f}" for x in stats['avg_credit_income']],
            textposition='auto',
            marker_color='#3B82F6'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=450,
        showlegend=False,
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Default Rate", row=1, col=1)
    fig.update_yaxes(title_text="Credit / Income Ratio", row=1, col=2)

    return fig

@st.cache_resource(show_spinner=False)
def create_employment_tenure_occupation_analysis(df):
    """
    Enhanced: Employment tenure x occupation vs default rate & income
    Occupations selected based on volume + default risk relevance
    """
    required_cols = [
        'DAYS_EMPLOYED',
        'OCCUPATION_TYPE',
        'AMT_INCOME_TOTAL',
        'TARGET'
    ]
    
    if not all(col in df.columns for col in required_cols):
        return go.Figure()

    df_copy = df.copy()

    # --------------------------------------------------
    # 1. Clean & engineer tenure
    # --------------------------------------------------
    df_copy = df_copy[df_copy['DAYS_EMPLOYED'] < 0]
    df_copy['EMP_TENURE_YEARS'] = (-df_copy['DAYS_EMPLOYED']) / 365

    df_copy['TENURE_BIN'] = pd.cut(
        df_copy['EMP_TENURE_YEARS'],
        bins=[0, 1, 3, 5, 10, 40],
        labels=['<1 yr', '1–3 yrs', '3–5 yrs', '5–10 yrs', '10+ yrs']
    )

    # --------------------------------------------------
    # 2. Occupation risk profiling (reused logic)
    # --------------------------------------------------
    occ_stats = df_copy.groupby('OCCUPATION_TYPE').agg(
        total=('TARGET', 'count'),
        default_rate=('TARGET', 'mean')
    ).reset_index()

    # Keep occupations with enough signal
    occ_stats = occ_stats[occ_stats['total'] > 500]

    # Select top occupations by risk relevance
    top_occ = (
        occ_stats
        .sort_values(['default_rate', 'total'], ascending=[False, False])
        .head(8)['OCCUPATION_TYPE']
    )

    df_copy = df_copy[df_copy['OCCUPATION_TYPE'].isin(top_occ)]

    # --------------------------------------------------
    # 3. Tenure × occupation aggregation
    # --------------------------------------------------
    stats = df_copy.groupby(['TENURE_BIN', 'OCCUPATION_TYPE']).agg(
        default_rate=('TARGET', 'mean'),
        avg_income=('AMT_INCOME_TOTAL', 'mean'),
        total=('TARGET', 'count')
    ).reset_index()

    # --------------------------------------------------
    # 4. Visualization
    # --------------------------------------------------
    fig = px.scatter(
        stats,
        x='avg_income',
        y='default_rate',
        size='total',
        color='TENURE_BIN',
        facet_col='OCCUPATION_TYPE',
        facet_col_wrap=4,
        title='Employment Tenure × Occupation: How Stability Modifies Risk',
        labels={
            'avg_income': 'Average Income',
            'default_rate': 'Default Rate',
            'TENURE_BIN': 'Employment Tenure'
        }
    )

    fig.update_layout(
        height=750,
        template='plotly_white'
    )

    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    return fig
