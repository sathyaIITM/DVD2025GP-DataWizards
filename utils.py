import streamlit as st

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Loan Risk Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def setup_custom_css():
    """Setup custom CSS styling"""
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.5rem;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #374151;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #E5E7EB;
        }
        .metric-container {
            background: #F8FAFC;
            border-radius: 10px;
            padding: 1.5rem;
            border-left: 4px solid #3B82F6;
            margin-bottom: 1rem;
        }
        .risk-low { color: #10B981; }
        .risk-medium { color: #F59E0B; }
        .risk-high { color: #EF4444; }
        .risk-veryhigh { color: #7C3AED; }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F3F4F6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3B82F6;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

def display_header():
    """Display dashboard header"""
    st.markdown('<p class="main-title">🏦 Loan Risk Analytics Dashboard</p>', unsafe_allow_html=True)
    # st.markdown("""
    # *Advanced loan application analysis with segmentation features and detailed Q&A*
    # """)

def display_kpi_metrics(df):
    """Display KPI metrics at the top of the dashboard"""
    st.header("📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", f"{len(df):,}")
    
    with col2:
        default_rate = df['TARGET'].mean() * 100 if 'TARGET' in df.columns else 0
        st.metric("Default Rate", f"{default_rate:.2f}%")
    
    with col3:
        if 'AMT_INCOME_TOTAL' in df.columns:
            avg_income = df['AMT_INCOME_TOTAL'].mean()
            st.metric("Avg Income", f"${avg_income:,.0f}")
    
    with col4:
        if 'AMT_CREDIT' in df.columns:
            avg_credit = df['AMT_CREDIT'].mean()
            st.metric("Avg Credit", f"${avg_credit:,.0f}")
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'AGE' in df.columns:
            avg_age = df['AGE'].mean()
            st.metric("Avg Age", f"{avg_age:.1f} years")
    
    with col2:
        if 'CREDIT_INCOME_RATIO' in df.columns:
            avg_ratio = df['CREDIT_INCOME_RATIO'].mean()
            st.metric("Avg Credit/Income", f"{avg_ratio:.2f}")
    
    with col3:
        if 'TARGET' in df.columns:
            default_count = df['TARGET'].sum()
            st.metric("Total Defaults", f"{default_count:,}")
    
    with col4:
        if 'CODE_GENDER' in df.columns:
            gender_dist = df['CODE_GENDER'].value_counts(normalize=True).get('M', 0) * 100
            st.metric("Male Applicants", f"{gender_dist:.1f}%")

def create_filters_sidebar(df):
    """Create sidebar filters and return filtered dataframe"""
    with st.sidebar:
        st.header("📊 Dashboard Controls")
        
        # Data source info
        st.markdown("### Data Source")
        st.info(f"**Dataset Loaded**")
        
        # Default rate info
        if 'TARGET' in df.columns:
            default_rate = df['TARGET'].mean() * 100
            st.metric("Overall Default Rate", f"{default_rate:.2f}%")
        
        # Filter controls
        st.markdown("### Filters")
        
        # Risk segment filter
        if 'RISK_SEGMENT' in df.columns:
            risk_options = ['All'] + sorted(df['RISK_SEGMENT'].dropna().unique().tolist())
            selected_risk = st.multiselect(
                "Risk Segment",
                options=risk_options,
                default=['All']
            )
            
            if 'All' not in selected_risk and selected_risk:
                df = df[df['RISK_SEGMENT'].isin(selected_risk)]
        
        # Age filter
        if 'AGE' in df.columns:
            min_age, max_age = int(df['AGE'].min()), int(df['AGE'].max())
            age_range = st.slider(
                "Age Range",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
            df = df[(df['AGE'] >= age_range[0]) & (df['AGE'] <= age_range[1])]
        
        # Income filter
        if 'AMT_INCOME_TOTAL' in df.columns:
            min_income = float(df['AMT_INCOME_TOTAL'].min())
            max_income = float(df['AMT_INCOME_TOTAL'].max())
            income_range = st.slider(
                "Income Range ($)",
                min_value=int(min_income),
                max_value=int(max_income),
                value=(int(min_income), int(max_income)),
                step=10000
            )
            df = df[(df['AMT_INCOME_TOTAL'] >= income_range[0]) & (df['AMT_INCOME_TOTAL'] <= income_range[1])]
        
        # Credit-to-income ratio filter
        if 'CREDIT_INCOME_RATIO' in df.columns:
            min_ratio = float(df['CREDIT_INCOME_RATIO'].min())
            max_ratio = float(df['CREDIT_INCOME_RATIO'].max())
            ratio_range = st.slider(
                "Credit/Income Ratio",
                min_value=float(min_ratio),
                max_value=float(max_ratio),
                value=(float(min_ratio), float(max_ratio)),
                step=0.5
            )
            df = df[(df['CREDIT_INCOME_RATIO'] >= ratio_range[0]) & (df['CREDIT_INCOME_RATIO'] <= ratio_range[1])]
        
        # Sample size control
        sample_size = st.slider(
            "Sample Size (%)",
            min_value=10,
            max_value=100,
            value=100
        )
        
        if sample_size < 100:
            df = df.sample(frac=sample_size/100, random_state=42)
        
        # Display filtered stats
        st.markdown("---")
        st.markdown("### Filtered Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Applications", f"{len(df):,}")
        with col2:
            if 'TARGET' in df.columns:
                filtered_rate = df['TARGET'].mean() * 100
                st.metric("Default Rate", f"{filtered_rate:.2f}%")
    
    return df