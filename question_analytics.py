import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_overview(preprocessed, app_data, prev_data, compact=False):
    """Display overview section"""
    if compact:
        # Compact version for main tab
        if 'overview' in preprocessed:
            overview = preprocessed['overview']
            st.metric("Previous Applications", f"{overview['total_prev_applications']:,}")
            st.metric("Application Columns", f"{overview['app_data_columns']}")
            st.metric("Previous App Columns", f"{overview['prev_data_columns']}")
    else:
        # Full version for Q&A tab
        if 'overview' in preprocessed:
            overview = preprocessed['overview']
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Applications", f"{overview['total_applications']:,}")
            with col2:
                st.metric("Overall Default Rate", f"{overview['default_rate']:.2f}%")
            with col3:
                st.metric("Previous Applications", f"{overview['total_prev_applications']:,}")
            with col4:
                default_count = app_data['TARGET'].sum()
                st.metric("Total Defaults", f"{default_count:,}")
            
            st.markdown("### Data Preview")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Current Applications Sample")
                st.write(f"Total columns: {overview['app_data_columns']}")
                st.dataframe(overview['app_data_head'], use_container_width=True)
                    
            with col2:
                st.subheader("Previous Applications Sample")
                st.write(f"Total columns: {overview['prev_data_columns']}")
                st.dataframe(overview['prev_data_head'], use_container_width=True)

def show_a1(preprocessed, app_data, prev_data):
    """Display A1: Age Groups & Default analysis"""

    st.header("👤 A1: How do age groups differ in default rates?")

    if 'age_analysis' not in preprocessed:
        st.warning("Age analysis data not available.")
        return

    age_data = preprocessed['age_analysis']

    col1, col2 = st.columns([2, 1])

    with col1:
        # Age group bar chart
        fig = px.bar(
            age_data['age_group_analysis'],
            x='Age Group',
            y='Default Rate (%)',
            title='Default Rate by Age Group',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)',
            hover_data=['Count', 'Mean Age']
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            height=500,
            xaxis_title="Age Group",
            yaxis_title="Default Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric(
            "Overall Correlation",
            f"{age_data['corr_age']:.3f}",
            "Negative = Older → Lower risk"
        )

        st.markdown("### Key Insights")
        st.info(f"""
        **Youngest Group (18-25):** {age_data['youngest_group_rate']:.1f}% default rate  
        **Oldest Group (60+):** {age_data['oldest_group_rate']:.1f}% default rate  
        **Difference:** {age_data['risk_diff_age']:.1f} percentage points
        """)

    # Age distribution
    st.subheader("Age Distribution Analysis")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Age Distribution (All Applicants)", "Age Distribution by Default Status"),
        specs=[[{"type": "histogram"}, {"type": "box"}]]
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=-app_data['DAYS_BIRTH'] / 365.25 if 'DAYS_BIRTH' in app_data.columns else [],
            nbinsx=20,
            name="All Applicants",
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    # Box plots
    if 'DAYS_BIRTH' in app_data.columns:
        default_ages = -app_data[app_data['TARGET'] == 1]['DAYS_BIRTH'] / 365.25
        non_default_ages = -app_data[app_data['TARGET'] == 0]['DAYS_BIRTH'] / 365.25

        fig.add_trace(
            go.Box(
                y=non_default_ages,
                name="Non-Default",
                marker_color='lightgreen'
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Box(
                y=default_ages,
                name="Default",
                marker_color='lightcoral'
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Age (Years)",
        yaxis_title="Count",
        xaxis2_title="Default Status",
        yaxis2_title="Age (Years)"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_a2(preprocessed, app_data, prev_data):
    """Display A2: Gender & Repayment analysis"""
    
    st.header("👫 A2: Does gender influence repayment behavior?")

    if 'gender_analysis' not in preprocessed:
        st.warning("Gender analysis data not available.")
        return

    gender_data = preprocessed['gender_analysis']

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        # Bar: counts
        fig.add_trace(go.Bar(
            x=gender_data['gender_default_analysis']['Gender'],
            y=gender_data['gender_default_analysis']['Count'],
            name='Number of Applicants',
            marker_color='lightblue',
            yaxis='y'
        ))

        # Line: default rate
        fig.add_trace(go.Scatter(
            x=gender_data['gender_default_analysis']['Gender'],
            y=gender_data['gender_default_analysis']['Default Rate (%)'],
            name='Default Rate',
            mode='lines+markers',
            marker=dict(size=10, color='red'),
            line=dict(color='red', width=3),
            yaxis='y2'
        ))

        fig.update_layout(
            title='Gender Analysis: Applications vs Default Rate',
            height=500,
            xaxis_title="Gender",
            yaxis=dict(
                title=dict(text="Number of Applicants", font=dict(color="blue")),
                tickfont=dict(color="blue")
            ),
            yaxis2=dict(
                title=dict(text="Default Rate (%)", font=dict(color="red")),
                tickfont=dict(color="red"),
                overlaying="y",
                side="right"
            ),
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        higher_risk = gender_data['higher_risk_gender']
        lower_risk = gender_data['lower_risk_gender']

        st.metric(
            f"Higher Risk Gender ({higher_risk['Gender']})",
            f"{higher_risk['Default Rate (%)']:.2f}%",
            f"{gender_data['risk_diff_gender']:.2f} pp higher"
        )

        st.metric(
            f"Lower Risk Gender ({lower_risk['Gender']})",
            f"{lower_risk['Default Rate (%)']:.2f}%",
            f"{lower_risk['Count']:,} applicants"
        )

        # Pie chart: gender distribution
        fig = px.pie(
            gender_data['gender_distribution'],
            values='Count',
            names='Gender',
            title='Gender Distribution',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender-Based Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        diff = gender_data['risk_diff_gender']
        if diff > 2:
            st.error(f"""
            **⚠️ Significant Gender Difference**  
            {higher_risk['Gender']} applicants have {diff:.2f} percentage points 
            higher default rate.
            """)
        elif diff > 1:
            st.warning("**⚠️ Moderate Gender Difference**")
        else:
            st.success("**✅ Minimal Gender Difference**")

    with col2:
        if diff > 2:
            st.info("""
            **Considerations:**
            - May require risk adjustments  
            - Must ensure fair-lending compliance  
            - Investigate underlying causes  
            """)
        else:
            st.info("""
            **Good News:**
            - Gender shows minimal risk impact  
            - Other variables should influence scoring  
            """)

    with st.expander("📊 Detailed Gender Statistics"):
        st.dataframe(
            gender_data['gender_default_analysis'].style.format({
                'Default Rate (%)': '{:.2f}%',
                'Count': '{:,}',
                'Defaults': '{:,}'
            }),
            use_container_width=True
        )

def show_a3(preprocessed, app_data, prev_data):
    """Display A3: Marital Status & Dependents analysis"""

    st.header("👨‍👩‍👧‍👦 A3: How do marital status and number of dependents relate to risk?")

    if 'family_analysis' not in preprocessed:
        st.warning("Family analysis data not available.")
        return

    family_data = preprocessed['family_analysis']

    # ------------------------------
    # 1. Marital Status
    # ------------------------------
    st.subheader("Marital Status Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.bar(
            family_data['marital_status_analysis'],
            x='Marital Status',
            y='Default Rate (%)',
            title='Default Rate by Marital Status',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)',
            hover_data=['Count']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        highest = family_data['highest_risk_marital']
        lowest = family_data['lowest_risk_marital']

        st.metric("Highest Risk Status", highest['Marital Status'], f"{highest['Default Rate (%)']:.1f}%")
        st.metric("Lowest Risk Status", lowest['Marital Status'], f"{lowest['Default Rate (%)']:.1f}%")
        st.metric("Risk Range", f"{family_data['risk_range_marital']:.1f} pp", "Difference")

    # ------------------------------
    # 2. Dependents
    # ------------------------------
    st.subheader("Number of Children Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.line(
            family_data['dependents_analysis'],
            x='Number of Children',
            y='Default Rate (%)',
            title='Default Rate by Number of Children',
            markers=True,
            line_shape='spline',
            text='Default Rate (%)'
        )
        fig.update_traces(
            textposition='top center',
            line=dict(width=3, color='darkblue'),
            marker=dict(size=10)
        )

        fig.add_trace(go.Bar(
            x=family_data['dependents_analysis']['Number of Children'],
            y=family_data['dependents_analysis']['Count'],
            name='Number of Applicants',
            opacity=0.3,
            marker_color='lightgray',
            yaxis='y2'
        ))

        fig.update_layout(
            height=400,
            yaxis2=dict(
                title="Number of Applicants",
                overlaying="y",
                side="right"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        optimal = family_data['optimal_children']
        st.metric("Optimal Number of Children", optimal, "Lowest default rate")
        st.metric("Correlation", f"{family_data['corr_children']:.3f}", "Positive = More children → Higher risk")
        st.info(f"**Insight:** Applicants with {optimal} children have the lowest default rate.")

    # ------------------------------
    # 3. Combined Summary
    # ------------------------------
    st.subheader("Combined Family Status Analysis")

    summary_df = pd.DataFrame({
        'Factor': ['Marital Status Range', 'Children Correlation', 'Optimal Children Count'],
        'Value': [
            f"{family_data['risk_range_marital']:.1f} percentage points",
            f"{family_data['corr_children']:.3f}",
            family_data['optimal_children']
        ],
        'Interpretation': [
            'Difference between highest and lowest risk marital statuses',
            'Relationship between number of children and default risk',
            'Number of children with lowest default rate'
        ]
    })

    st.dataframe(
        summary_df.style.set_properties(
            **{'background-color': '#f0f2f6', 'color': 'black'}
        ),
        use_container_width=True
    )

    # ------------------------------
    # 4. Risk Assessment
    # ------------------------------
    st.markdown("### Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:
        if family_data['risk_range_marital'] > 5:
            st.error("**⚠️ High Marital Status Impact**")
        elif family_data['risk_range_marital'] > 2:
            st.warning("**⚠️ Moderate Marital Status Impact**")
        else:
            st.success("**✅ Low Marital Status Impact**")

    with col2:
        corr = abs(family_data['corr_children'])
        if corr > 0.1:
            st.error("**⚠️ Significant Children Impact**")
        elif corr > 0.05:
            st.warning("**⚠️ Moderate Children Impact**")
        else:
            st.success("**✅ Low Children Impact**")

    # ------------------------------
    # 5. Details
    # ------------------------------
    with st.expander("📊 Detailed Family Statistics"):
        tab1, tab2 = st.tabs(["Marital Status", "Number of Children"])

        with tab1:
            st.dataframe(
                family_data['marital_status_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

        with tab2:
            st.dataframe(
                family_data['dependents_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

def show_b4(preprocessed, app_data, prev_data):
    '''render_income_level_default'''
    if 'income_level_analysis' not in preprocessed:
        st.warning("Income level analysis not available.")
        return

    income_data = preprocessed['income_level_analysis']

    st.header("💰 B4: How does income level relate to default?")

    # --------------------------------------------------------------
    # Income Quintiles
    # --------------------------------------------------------------
    st.subheader("Income Quintile Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        df = income_data['income_quintile_analysis']
        fig = px.bar(
            df,
            x='Income Quintile',
            y='Default Rate (%)',
            title='Default Rate by Income Quintiles',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)',
            hover_data=['Mean Income', 'Count']
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            height=400,
            xaxis_title="Income Quintile (Q1 = Lowest, Q5 = Highest)",
            yaxis_title="Default Rate (%)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Lowest Income (Q1)", f"{income_data['lowest_income_rate']:.1f}%", "Highest risk")
        st.metric("Highest Income (Q5)", f"{income_data['highest_income_rate']:.1f}%", "Lowest risk")
        st.metric("Risk Difference", f"{income_data['income_risk_diff']:.1f} pp", "Q1 vs Q5")
        st.metric("Correlation", f"{income_data['corr_income']:.3f}", "Negative = Higher income → Lower risk")

    # --------------------------------------------------------------
    # Income Category Analysis
    # --------------------------------------------------------------
    st.subheader("Income Category Analysis")

    fig = px.bar(
        income_data['income_category_analysis'],
        x='Income Category',
        y='Default Rate (%)',
        title='Default Rate by Income Categories',
        color='Default Rate (%)',
        color_continuous_scale='RdYlGn_r',
        text='Default Rate (%)',
        hover_data=['Mean Income', 'Count']
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------
    # Income Risk Analysis Section
    # --------------------------------------------------------------
    st.subheader("Income Risk Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=abs(income_data['corr_income']) * 100,
            title={'text': "Income Predictive Power"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "darkgray"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        color = "crimson" if income_data['income_risk_diff'] > 5 else "orange" if income_data['income_risk_diff'] > 2 else "green"
        fig = go.Figure([go.Bar(
            x=['Lowest vs Highest Income'],
            y=[income_data['income_risk_diff']],
            marker_color=[color]
        )])
        fig.update_layout(title='Risk Gap: Lowest vs Highest Income', height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("### 📊 Key Insights")
        corr = income_data['corr_income']

        if corr < -0.2:
            st.success("**Strong Income Effect — Higher income relates to lower default risk.**")
        elif corr < -0.1:
            st.info("**Moderate Income Effect — Income meaningfully influences default risk.**")
        else:
            st.warning("**Weak Income Effect — Income has limited predictive power.**")

    # --------------------------------------------------------------
    # Business Implications
    # --------------------------------------------------------------
    st.subheader("Business Implications")

    diff = income_data['income_risk_diff']
    if diff > 5:
        st.info("""
        - Income-based risk adjustment  
        - Target high-income applicants  
        - Stronger income verification  
        - Income-tiered product design  
        """)
    elif diff > 2:
        st.info("""
        - Include income moderately in risk scoring  
        - Segment income groups  
        - Consider income growth trajectory  
        """)
    else:
        st.info("""
        - Income has mild impact  
        - Use alongside stronger predictors  
        - Consider broader financial health  
        """)

    # --------------------------------------------------------------
    # Detailed Stats
    # --------------------------------------------------------------
    with st.expander("📊 Detailed Income Statistics"):
        tab1, tab2 = st.tabs(["Income Quintiles", "Income Categories"])

        with tab1:
            st.dataframe(income_data['income_quintile_analysis'])

        with tab2:
            st.dataframe(income_data['income_category_analysis'])

def show_b5(preprocessed, app_data, prev_data):
    '''render_credit_amount_terms'''

    if 'credit_terms_analysis' not in preprocessed:
        st.warning("Credit terms analysis not available.")
        return

    terms_data = preprocessed['credit_terms_analysis']

    st.header("💳 B5: How do credit amount, annuity, and loan term relate to repayment?")

    # --------------------------------------------------------------
    # Credit Amount Analysis
    # --------------------------------------------------------------
    st.subheader("Credit Amount Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        df = terms_data['credit_amount_analysis']
        fig = px.bar(
            df,
            x='Credit Quintile',
            y='Default Rate (%)',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)',
            hover_data=['Mean Credit', 'Count']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Correlation", f"{terms_data['corr_credit_amount']:.3f}")

        highest = df.loc[df['Default Rate (%)'].idxmax()]
        lowest = df.loc[df['Default Rate (%)'].idxmin()]

        st.metric("Highest Risk Amount", highest['Credit Quintile'], f"{highest['Default Rate (%)']:.1f}%")
        st.metric("Lowest Risk Amount", lowest['Credit Quintile'], f"{lowest['Default Rate (%)']:.1f}%")

    # --------------------------------------------------------------
    # Annuity Analysis
    # --------------------------------------------------------------
    st.subheader("Annuity Payment Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        df = terms_data['annuity_analysis']
        fig = px.bar(
            df,
            x='Annuity Quintile',
            y='Default Rate (%)',
            text='Default Rate (%)',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r'
        )
        fig.update_traces(texttemplate='%{text:.1f}%')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Correlation", f"{terms_data['corr_annuity']:.3f}")

        highest = df.loc[df['Default Rate (%)'].idxmax()]
        lowest = df.loc[df['Default Rate (%)'].idxmin()]
        st.metric("Highest Risk Annuity", highest['Annuity Quintile'], f"{highest['Default Rate (%)']:.1f}%")
        st.metric("Lowest Risk Annuity", lowest['Annuity Quintile'], f"{lowest['Default Rate (%)']:.1f}%")

    # --------------------------------------------------------------
    # Loan Term Analysis
    # --------------------------------------------------------------
    df = terms_data['loan_term_analysis']
    if not df.empty:
        st.subheader("Loan Term Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            fig = px.bar(
                df,
                x='Loan Term',
                y='Default Rate (%)',
                text='Default Rate (%)',
                color='Default Rate (%)',
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_traces(texttemplate='%{text:.1f}%')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            corr = terms_data['corr_loan_term']
            st.metric("Correlation", f"{corr:.3f}" if not np.isnan(corr) else "N/A")

            highest = df.loc[df['Default Rate (%)'].idxmax()]
            lowest = df.loc[df['Default Rate (%)'].idxmin()]
            st.metric("Highest Risk Term", highest['Loan Term'], f"{highest['Default Rate (%)']:.1f}%")
            st.metric("Lowest Risk Term", lowest['Loan Term'], f"{lowest['Default Rate (%)']:.1f}%")

    # --------------------------------------------------------------
    # Combined Correlation Table
    # --------------------------------------------------------------
    st.subheader("Combined Credit Terms Risk Analysis")

    corr_df = pd.DataFrame({
        'Factor': ['Credit Amount', 'Annuity Payment', 'Loan Term'],
        'Correlation': [
            terms_data['corr_credit_amount'],
            terms_data['corr_annuity'],
            terms_data['corr_loan_term']
        ]
    })

    st.dataframe(corr_df)

    # --------------------------------------------------------------
    # Business Insights
    # --------------------------------------------------------------
    st.subheader("Business Implications")

    col1, col2 = st.columns(2)

    with col1:
        corr = terms_data['corr_credit_amount']
        if abs(corr) > 0.2:
            st.error("Strong Credit Amount Impact — Higher credit leads to higher risk.")
        elif abs(corr) > 0.1:
            st.warning("Moderate Credit Amount Impact — Should be included in scoring.")
        else:
            st.success("Weak Credit Amount Impact — Low predictive power.")

    with col2:
        if terms_data['corr_credit_amount'] > 0 and terms_data['corr_annuity'] > 0:
            st.info("""
            - Strengthen income verification  
            - Require collateral for large loans  
            - Graduated approval levels  
            """)
        else:
            st.info("""
            - Use credit terms as moderate risk factors  
            - Combine with income evaluation  
            - Monitor concentration risk  
            """)

    # --------------------------------------------------------------
    # Detailed Stats
    # --------------------------------------------------------------
    with st.expander("📊 Detailed Credit Terms Statistics"):
        tab1, tab2, tab3 = st.tabs(["Credit Amount", "Annuity", "Loan Term"])
        with tab1:
            st.dataframe(terms_data['credit_amount_analysis'])
        with tab2:
            st.dataframe(terms_data['annuity_analysis'])
        with tab3:
            st.dataframe(terms_data['loan_term_analysis'])

def show_b6(preprocessed, app_data, prev_data):
    '''render_income_credit_ratios'''

    if 'q1' not in preprocessed:
        st.warning("Ratio analysis not available.")
        return

    q1 = preprocessed['q1']

    st.header("📊 B6: Income & Credit Ratios — Do higher ratios signal higher risk?")

    # --------------------------------------------------------------
    # Top Row — Ratio Charts
    # --------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        df = q1['credit_income_analysis']
        fig = px.bar(
            df,
            x='Quintile',
            y='Default Rate (%)',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Credit-to-Income Correlation", f"{q1['corr_credit_income']:.3f}")
        st.metric("Risk Difference (Q5–Q1)",
                  f"{q1['high_credit_income'] - q1['low_credit_income']:.1f} pp",
                  f"{q1['high_credit_income']:.1f}% vs {q1['low_credit_income']:.1f}%")

    with col2:
        df = q1['annuity_income_analysis']
        fig = px.bar(
            df,
            x='Quintile',
            y='Default Rate (%)',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)'
        )
        fig.update_traces(texttemplate='%{text:.1f}%')
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Annuity-to-Income Correlation", f"{q1['corr_annuity_income']:.3f}")
        st.metric("Risk Difference (Q5–Q1)",
                  f"{q1['high_annuity_income'] - q1['low_annuity_income']:.1f} pp",
                  f"{q1['high_annuity_income']:.1f}% vs {q1['low_annuity_income']:.1f}%")

    # --------------------------------------------------------------
    # Insights Section
    # --------------------------------------------------------------
    st.subheader("Ratio Analysis Insights")

    col1, col2 = st.columns(2)

    with col1:
        corr = q1['corr_credit_income']
        if corr > 0.2:
            st.error("""
            Strong Warning Signal — High credit-to-income ratios sharply increase risk.
            """)
        elif corr > 0.1:
            st.warning("""
            Moderate Warning — Higher ratios correlate with higher risk.
            """)
        else:
            st.success("""
            Limited Predictive Power — Credit-to-income ratio is a secondary factor.
            """)

    with col2:
        corr = q1['corr_annuity_income']
        if corr > 0.2:
            st.error("Strong Warning Signal — High annuity burdens increase risk.")
        elif corr > 0.1:
            st.warning("Moderate Warning — Ratios influence affordability and risk.")
        else:
            st.success("Limited Predictive Power — Annuity ratios are supportive metrics.")

    # --------------------------------------------------------------
    # Comparison Table
    # --------------------------------------------------------------
    st.subheader("Risk Comparison & Business Rules")

    comparison = {
        'Metric': ['Credit-to-Income', 'Annuity-to-Income'],
        'Correlation': [q1['corr_credit_income'], q1['corr_annuity_income']],
        'Q1 Risk (%)': [q1['low_credit_income'], q1['low_annuity_income']],
        'Q5 Risk (%)': [q1['high_credit_income'], q1['high_annuity_income']]
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(comparison))

def show_c7(preprocessed, app_data, prev_data):
    st.header("💼 C7: Does employment tenure affect default probability?")

    # ------------------------------------------
    # Top Row — Tenure Categories
    # ------------------------------------------
    st.subheader("Employment Tenure Categories")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.bar(
            preprocessed['tenure_category_analysis'],
            x='Tenure Category',
            y='Default Rate (%)',
            title='Default Rate by Employment Tenure',
            color='Default Rate (%)',
            color_continuous_scale='RdYlGn_r',
            text='Default Rate (%)',
            hover_data=['Mean Years', 'Count']
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            height=400,
            xaxis_title="Employment Tenure",
            yaxis_title="Default Rate (%)",
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Shortest Tenure (< 6 months)",
                  f"{preprocessed['shortest_tenure']:.1f}%", "Highest risk")
        st.metric("Longest Tenure (10+ years)",
                  f"{preprocessed['longest_tenure']:.1f}%", "Lowest risk")
        st.metric("Risk Difference",
                  f"{preprocessed['diff_tenure']:.1f} pp",
                  f"{preprocessed['shortest_tenure']:.1f}% vs {preprocessed['longest_tenure']:.1f}%")
        st.metric("Correlation",
                  f"{preprocessed['corr_tenure']:.3f}",
                  "Negative = Longer tenure → Lower risk")

    # ------------------------------------------
    # Middle Row — Tenure Quintiles
    # ------------------------------------------
    st.subheader("Employment Tenure Quintiles")

    fig = px.bar(
        preprocessed['tenure_quintile_analysis'],
        x='Quintile',
        y='Default Rate (%)',
        title='Default Rate by Employment Tenure Quintiles',
        color='Default Rate (%)',
        color_continuous_scale='RdYlGn_r',
        text='Default Rate (%)',
        hover_data=['Mean Years', 'Count']
    )
    fig.update_traces(
        texttemplate='%{text:.1f}%',
        textposition='outside'
    )
    fig.update_layout(
        height=400,
        xaxis_title="Tenure Quintile (Q1 = Shortest, Q5 = Longest)",
        yaxis_title="Default Rate (%)",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------
    # Bottom Row — Insights
    # ------------------------------------------
    st.subheader("Employment Stability Analysis")

    col1, col2 = st.columns(2)

    # Risk Assessment
    with col1:
        st.markdown("### Risk Assessment")

        if preprocessed['corr_tenure'] < -0.2:
            st.error("""
            **⚠️ Strong Employment Stability Effect**  
            Employment tenure is a powerful predictor of default risk.

            **Key Findings:**
            - New employees (<6 months): Very high risk  
            - Long-term employees (>10 years): Very low risk  
            - Clear progression as tenure increases  
            """)
        elif preprocessed['corr_tenure'] < -0.1:
            st.warning("""
            **⚠️ Moderate Employment Stability Effect**  
            Employment tenure shows meaningful correlation with default risk.

            **Key Findings:**
            - Shorter tenure correlates with higher risk  
            - Stability improves with longer employment  
            - Consider as a moderate risk factor  
            """)
        else:
            st.success("""
            **✅ Weak Employment Stability Effect**  
            Employment tenure shows limited correlation with default risk.

            **Key Findings:**
            - Minimal difference across tenure groups  
            - Focus on other employment factors  
            - Consider industry-specific patterns  
            """)

    # Business Implications
    with col2:
        st.markdown("### Business Implications")

        if preprocessed['diff_tenure'] > 10:
            st.info("""
            **For High Risk (Short Tenure):**
            1. Additional employment verification  
            2. Require employment history  
            3. Consider probation period status  
            4. Higher scrutiny for job changes  

            **For Low Risk (Long Tenure):**
            1. Streamlined approval process  
            2. Consider tenure as positive factor  
            3. Loyalty program opportunities  
            4. Preferred customer status  
            """)
        elif preprocessed['diff_tenure'] > 5:
            st.info("""
            **Risk-Based Approach:**
            1. Include tenure in scoring models  
            2. Differentiate by tenure brackets  
            3. Monitor job stability patterns  
            4. Consider industry stability  
            """)
        else:
            st.info("""
            **Standard Approach:**
            1. Use tenure as secondary factor  
            2. Combine with income verification  
            3. Consider overall employment profile  
            4. Monitor for industry trends  
            """)

    # ------------------------------------------
    # Additional Insights
    # ------------------------------------------
    st.markdown("### Additional Insights")

    col1, col2, col3 = st.columns(3)

    # Critical threshold
    with col1:
        if preprocessed['tenure_category_analysis'].iloc[0]['Default Rate (%)'] > 15:
            st.metric("Critical Risk Period", "First 6 months", "Very high default risk")
        else:
            st.metric("Stabilization Point", "2–5 years", "Risk significantly reduces")

    # Optimal tenure range
    with col2:
        optimal = preprocessed['tenure_category_analysis'].loc[
            preprocessed['tenure_category_analysis']['Default Rate (%)'].idxmin()
        ]
        st.metric("Optimal Tenure Range", optimal['Tenure Category'],
                  f"{optimal['Default Rate (%)']:.1f}% default rate")

    # Risk reduction rate
    with col3:
        if len(preprocessed['tenure_category_analysis']) >= 2:
            reduction = (
                preprocessed['tenure_category_analysis'].iloc[0]['Default Rate (%)'] -
                preprocessed['tenure_category_analysis'].iloc[-1]['Default Rate (%)']
            ) / 10
            st.metric("Annual Risk Reduction", f"{reduction:.1f} pp/year",
                      "With increasing tenure")

    # ------------------------------------------
    # Detailed statistics
    # ------------------------------------------
    with st.expander("📊 Detailed Tenure Statistics"):
        tab1, tab2 = st.tabs(["Tenure Categories", "Tenure Quintiles"])

        with tab1:
            st.dataframe(
                preprocessed['tenure_category_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Mean Years': '{:.1f} years',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

        with tab2:
            st.dataframe(
                preprocessed['tenure_quintile_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Mean Years': '{:.1f} years',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

def show_c8(preprocessed, app_data, prev_data):
    st.header("👔 C8: Are specific employment types or occupations higher risk?")

    # ------------------------------------------
    # Employment Types Analysis
    # ------------------------------------------
    st.subheader("Employment Types Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.bar(
            preprocessed['income_type_analysis'],
            x='Income Type',
            y='Default Rate (%)',
            title='Default Rate by Employment Type',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)',
            hover_data=['Count', 'Defaults']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=500,
            xaxis_title="Employment/Income Type",
            yaxis_title="Default Rate (%)",
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top 3 Riskiest Employment Types")
        for _, row in preprocessed['top_risk_income'].iterrows():
            st.metric(
                row['Income Type'],
                f"{row['Default Rate (%)']:.1f}%",
                f"{row['Count']:,} applicants"
            )

        if preprocessed['risk_diff_income']:
            st.metric("Risk Range", f"{preprocessed['risk_diff_income']:.1f} pp",
                      "Highest vs Lowest")

    # ------------------------------------------
    # Occupations Analysis
    # ------------------------------------------
    st.subheader("Occupations Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        top_occ = preprocessed['occupation_analysis'].head(15)
        fig = px.bar(
            top_occ,
            x='Occupation Type',
            y='Default Rate (%)',
            title='Default Rate by Occupation Type (Top 15)',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)',
            hover_data=['Count', 'Defaults']
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(
            height=500,
            xaxis_title="Occupation Type",
            yaxis_title="Default Rate (%)",
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top 3 Riskiest Occupations")
        for _, row in preprocessed['top_risk_occ'].iterrows():
            display_name = row['Occupation Type'][:20] + \
                ("..." if len(row['Occupation Type']) > 20 else "")
            st.metric(display_name,
                      f"{row['Default Rate (%)']:.1f}%",
                      f"{row['Count']:,} applicants")

        if preprocessed['risk_diff_occ']:
            st.metric("Risk Range", f"{preprocessed['risk_diff_occ']:.1f} pp",
                      "Highest vs Lowest")

    # ------------------------------------------
    # Risk Insights
    # ------------------------------------------
    st.subheader("Employment Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Employment Type Risk Assessment")

        r = preprocessed['risk_range_income']

        if r and r > 10:
            st.error("""
            **⚠️ Very High Risk Variation**  
            Employment types show extremely wide risk variation.
            """)
        elif r and r > 5:
            st.warning("""
            **⚠️ High Risk Variation**  
            Significant differences across employment types.
            """)
        elif r:
            st.info("""
            **ℹ️ Moderate Risk Variation**  
            Noticeable differences across employment types.
            """)

    with col2:
        st.markdown("### Occupation Risk Assessment")

        r = preprocessed['risk_diff_occ']

        if r and r > 15:
            st.error("""
            **⚠️ Extreme Occupation Risk Variation**  
            Massive risk differences across occupations.
            """)
        elif r and r > 8:
            st.warning("""
            **⚠️ Significant Occupation Risk Variation**  
            Occupations are strong predictors of risk.
            """)
        elif r:
            st.info("""
            **ℹ️ Moderate Occupation Risk Variation**  
            Occupations show meaningful differences.
            """)

    # ------------------------------------------
    # Business Recommendations
    # ------------------------------------------
    st.subheader("Business Recommendations")

    r = preprocessed['risk_range_income']

    if r and r > 5:
        st.info("""
        **For High-Risk Employment Types:**
        - Require stronger income verification  
        - Manual review for risky categories  
        - Lower starting credit limits  
        - Close repayment monitoring  

        **For Low-Risk Employment Types:**
        - Streamlined approval  
        - Higher initial credit limits  
        - Better pricing or terms  
        - Loyalty & retention programs  
        """)
    else:
        st.info("""
        **Standard Employment Assessment:**
        - Combine employment type with other risk factors  
        - Monitor industry-wide shifts  
        - Emphasize employment stability  
        - Review risk models periodically  
        """)

    # ------------------------------------------
    # Detailed Statistics
    # ------------------------------------------
    with st.expander("📊 Detailed Employment Statistics"):
        tab1, tab2 = st.tabs(["Employment Types", "Occupations"])

        with tab1:
            st.dataframe(
                preprocessed['income_type_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}',
                    'Defaults': '{:,}'
                }),
                use_container_width=True
            )

        with tab2:
            st.dataframe(
                preprocessed['occupation_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}',
                    'Defaults': '{:,}'
                }),
                use_container_width=True
            )

def show_d9(preprocessed, app_data, prev_data):
    """Display D9: Previous Loan Outcomes"""
    st.header("📋 D9: How do previous loan outcomes relate to current default?")

    if 'q4' not in preprocessed:
        st.warning("Previous loan outcome data unavailable.")
        return

    q4 = preprocessed['q4']

    # ------------------------------
    # Previous Application History
    # ------------------------------
    st.subheader("Previous Application History Impact")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            q4['status_analysis'],
            x='Category',
            y='default_rate',
            title='Previous Applications vs No History',
            color='Category',
            color_discrete_map={
                'No Previous Applications': '#4ECDC4',
                'Has Previous Applications': '#FF6B6B'
            },
            text='default_rate',
            labels={'default_rate': 'Default Rate (%)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Default Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Difference",
            f"{q4['has_prev_rate'] - q4['no_prev_rate']:.1f} pp",
            f"{q4['has_prev_rate']:.1f}% vs {q4['no_prev_rate']:.1f}%"
        )

    with col2:
        fig = px.bar(
            q4['approved_analysis'],
            x='Category',
            y='default_rate',
            title='Previous Approval Status Impact',
            color='Category',
            color_discrete_map={
                'Never Approved': '#FF6B6B',
                'Previously Approved': '#4ECDC4'
            },
            text='default_rate',
            labels={'default_rate': 'Default Rate (%)'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Default Rate (%)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            "Approval Advantage",
            f"{q4['never_approved_rate'] - q4['prev_approved_rate']:.1f} pp",
            f"{q4['prev_approved_rate']:.1f}% vs {q4['never_approved_rate']:.1f}%"
        )

    # ------------------------------
    # Outcome Pattern Analysis
    # ------------------------------
    st.subheader("Previous Outcome Pattern Analysis")

    fig = px.bar(
        q4['outcome_analysis'],
        x='Previous Outcome Pattern',
        y='Default Rate (%)',
        title='Detailed Previous Loan Outcome Patterns',
        color='Default Rate (%)',
        color_continuous_scale='Reds',
        text='Default Rate (%)',
        hover_data=['Count']
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        height=500, xaxis_title="Previous Outcome Pattern",
        yaxis_title="Default Rate (%)",
        xaxis_tickangle=-45, showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Risk Indicators
    # ------------------------------
    st.subheader("Credit History Risk Analysis")

    col1, col2, col3 = st.columns(3)

    # Previous application impact
    with col1:
        prev_app_impact = q4['has_prev_rate'] - q4['no_prev_rate']
        if prev_app_impact > 3:
            st.error(f"""
            **⚠️ High Previous Application Risk**  
            Risk increases by {prev_app_impact:.1f} pp.
            """)
        elif prev_app_impact > 1:
            st.warning(f"""
            **⚠️ Moderate Previous Application Risk**  
            Increase of {prev_app_impact:.1f} pp.
            """)
        else:
            st.success(f"""
            **✅ Low Previous Application Impact**  
            Only {prev_app_impact:.1f} pp difference.
            """)

    # Approval status impact
    with col2:
        approval_impact = q4['never_approved_rate'] - q4['prev_approved_rate']
        if approval_impact > 3:
            st.success(f"""
            **✅ Strong Positive Signal**  
            Approval reduces risk by {approval_impact:.1f} pp.
            """)
        elif approval_impact > 1:
            st.info(f"""
            **ℹ️ Moderate Positive Signal**  
            Reduction of {approval_impact:.1f} pp.
            """)
        else:
            st.warning(f"""
            **⚠️ Weak Approval Signal**  
            Impact: {approval_impact:.1f} pp.
            """)

    # Best predictors
    with col3:
        highest_risk = q4['outcome_analysis'].iloc[0]
        lowest_risk = q4['outcome_analysis'].iloc[-1]

        st.metric(
            "Highest Risk Pattern",
            highest_risk['Previous Outcome Pattern'][:15] + "...",
            f"{highest_risk['Default Rate (%)']:.1f}%"
        )
        st.metric(
            "Lowest Risk Pattern",
            lowest_risk['Previous Outcome Pattern'][:15] + "...",
            f"{lowest_risk['Default Rate (%)']:.1f}%"
        )

    # ------------------------------
    # Business Recommendations
    # ------------------------------
    st.subheader("Business Recommendations")

    if q4['has_prev_rate'] > q4['no_prev_rate'] + 2:
        st.info("""
        **Applicants with previous credit attempts should undergo:**
        - Deeper review of historical outcomes  
        - Pattern-based evaluation  
        - Time since last application analysis  
        - Frequency monitoring  
        """)
    else:
        st.info("""
        **Use standard credit history review protocols:**
        - Holistic multi-factor evaluation  
        - Emphasis on recent outcomes  
        - Identification of trend improvements  
        """)

    # ------------------------------
    # Expandable: Detailed Stats
    # ------------------------------
    with st.expander("📊 Detailed Previous Loan Statistics"):
        st.dataframe(
            q4['outcome_analysis'].style.format({
                'Default Rate (%)': '{:.2f}%',
                'Count': '{:,}'
            }),
            use_container_width=True
        )

def show_d10(preprocessed, app_data, prev_data):
    """Display D10: Credit Bureau Enquiries"""
    st.header("🔍 D10: Does credit bureau enquiry frequency correlate with higher risk?")

    if 'q5' not in preprocessed:
        st.warning("Enquiry data unavailable.")
        return

    q5 = preprocessed['q5']

    # ------------------------------
    # Total Enquiry Analysis
    # ------------------------------
    st.subheader("Credit Bureau Enquiry Analysis")

    col1, col2 = st.columns([3, 1])

    with col1:
        if not q5['total_enquiry_analysis'].empty:
            fig = px.bar(
                q5['total_enquiry_analysis'],
                x='Total Enquiries',
                y='Default Rate (%)',
                title='Default Rate by Credit Bureau Enquiry Status',
                color='Default Rate (%)',
                color_continuous_scale='Reds',
                text='Default Rate (%)',
                hover_data=['Count']
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, xaxis_title="Credit Bureau Enquiry Status",
                              yaxis_title="Default Rate (%)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No enquiry data available.")

    with col2:
        if q5['corr_total'] is not None:
            st.metric(
                "Correlation",
                f"{q5['corr_total']:.3f}",
                "Positive → Higher risk" if q5['corr_total'] > 0 else "Negative → Lower risk"
            )

        if q5['zero_enq_rate'] is not None:
            st.metric("No Enquiries", f"{q5['zero_enq_rate']:.1f}%", "Default rate")

        if q5['non_zero_enq_rate'] is not None:
            diff = q5['non_zero_enq_rate'] - q5['zero_enq_rate']
            st.metric("With Enquiries", f"{q5['non_zero_enq_rate']:.1f}%", f"{diff:+.1f} pp difference")

    # ------------------------------
    # Enquiry Timing Analysis
    # ------------------------------
    st.subheader("Enquiry Timing Analysis")

    timing_cols = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR'
    ]

    if all(c in app_data.columns for c in timing_cols):
        timing_data = app_data[['TARGET'] + timing_cols].copy()
        timing_analysis = []

        for col in timing_cols:
            grouped = timing_data.groupby(timing_data[col] > 0)['TARGET'].agg([
                ('default_rate', lambda x: (x.mean() * 100)),
                ('count', 'count')
            ]).reset_index()

            if len(grouped) == 2:
                has_enq = grouped[grouped[col] == True]
                no_enq = grouped[grouped[col] == False]

                timing_analysis.append({
                    'Timeframe': col.replace('AMT_REQ_CREDIT_BUREAU_', '').title(),
                    'With Enquiries Rate': has_enq['default_rate'].values[0],
                    'No Enquiries Rate': no_enq['default_rate'].values[0],
                    'Difference': has_enq['default_rate'].values[0] - no_enq['default_rate'].values[0],
                })

        if timing_analysis:
            df = pd.DataFrame(timing_analysis)

            fig = go.Figure([
                go.Bar(name='No Enquiries', x=df['Timeframe'], y=df['No Enquiries Rate']),
                go.Bar(name='With Enquiries', x=df['Timeframe'], y=df['With Enquiries Rate'])
            ])
            fig.update_layout(
                title='Default Rate by Enquiry Timing',
                height=400, barmode='group',
                xaxis_title="Timeframe Before Application",
                yaxis_title="Default Rate (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Risk Analysis
    # ------------------------------
    st.subheader("Enquiry Frequency Risk Analysis")

    if q5['corr_total'] is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            if abs(q5['corr_total']) > 0.15:
                st.error(f"**⚠️ Strong Correlation**\nCorrelation: {q5['corr_total']:.3f}")
            elif abs(q5['corr_total']) > 0.05:
                st.warning(f"**⚠️ Moderate Correlation**\nCorrelation: {q5['corr_total']:.3f}")
            else:
                st.success(f"**✅ Weak Correlation**\nCorrelation: {q5['corr_total']:.3f}")

        with col2:
            if q5['corr_total'] > 0.05:
                st.error("**⚠️ Risk Increase**\nMore enquiries → Higher risk.")
            elif q5['corr_total'] < -0.05:
                st.success("**✅ Risk Decrease**\nMore enquiries → Lower risk.")
            else:
                st.info("**ℹ️ Neutral Impact**\nMinimal predictive power.")

        with col3:
            if q5['zero_enq_rate'] is not None and q5['non_zero_enq_rate'] is not None:
                diff = abs(q5['non_zero_enq_rate'] - q5['zero_enq_rate'])
                label = "High Impact" if diff > 5 else "Moderate Impact" if diff > 2 else "Low Impact"
                st.metric(label, f"{diff:.1f} pp", "Default rate difference")

    # ------------------------------
    # Recommendations
    # ------------------------------
    st.subheader("Business Recommendations")

    if q5['corr_total'] is not None and q5['corr_total'] > 0.1:
        st.info("""
        **High enquiry frequency → Higher risk. Recommended:**
        - Manual review  
        - Check enquiry timing  
        - Investigate credit-seeking reasons  
        - Use conservative credit limits  
        """)
    elif q5['corr_total'] is not None and q5['corr_total'] < -0.1:
        st.info("""
        **Enquiries may indicate responsible credit activity.**
        Recommended:
        - Moderate weightage  
        - Pattern recognition  
        - Corroborate with bureau scores  
        """)
    else:
        st.info("""
        **Neutral impact. Use enquiries as a secondary signal.**
        - Consider timing and intent  
        - Combine with other factors  
        """)

    # ------------------------------
    # Detailed Stats
    # ------------------------------
    with st.expander("📊 Detailed Enquiry Statistics"):
        if not q5['total_enquiry_analysis'].empty:
            st.dataframe(
                q5['total_enquiry_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

def show_e11(preprocessed, app_data, prev_data):
    """Display E11: Loan Types (Cash vs Revolving) analysis"""
    st.header("💸 E11: Are cash loans riskier than revolving credits?")
    
    if 'loan_type_analysis' in preprocessed:
        loan_data = preprocessed['loan_type_analysis']
        
        # Top row: Contract type analysis
        st.subheader("Loan Type Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = px.bar(
                loan_data['contract_type_analysis'],
                x='Contract Type',
                y='Default Rate (%)',
                title='Default Rate by Loan Type',
                color='Default Rate (%)',
                color_continuous_scale='Reds',
                text='Default Rate (%)',
                hover_data=['Count']
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            fig.update_layout(
                height=400,
                xaxis_title="Loan Type",
                yaxis_title="Default Rate (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            highest_risk = loan_data['highest_risk_type']
            st.metric(
                "Highest Risk Type",
                highest_risk['Contract Type'],
                f"{highest_risk['Default Rate (%)']:.1f}%"
            )
            
            st.metric(
                "Risk Difference",
                f"{loan_data['risk_diff_contract']:.1f} pp",
                f"{loan_data['riskier_type']} is riskier"
            )
            
            # Distribution pie chart
            fig = px.pie(
                loan_data['contract_distribution'],
                values='Count',
                names='Contract Type',
                title='Loan Type Distribution',
                hole=0.4
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        st.subheader("Loan Type Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if loan_data['risk_diff_contract'] > 5:
                st.error(f"""
                **⚠️ Significant Loan Type Difference**  
                {loan_data['riskier_type']} loans have {loan_data['risk_diff_contract']:.1f} pp higher default rate.
                
                **Implications:**
                - Major factor in risk assessment
                - Different pricing strategies needed
                - Separate risk models for each type
                """)
            elif loan_data['risk_diff_contract'] > 2:
                st.warning(f"""
                **⚠️ Moderate Loan Type Difference**  
                Noticeable risk differences between loan types.
                
                **Implications:**
                - Consider in risk scoring
                - Different approval criteria
                - Monitor type-specific trends
                """)
            else:
                st.success("""
                **✅ Minimal Loan Type Difference**  
                Loan types show similar risk profiles.
                
                **Implications:**
                - Standardized risk assessment
                - Focus on other factors
                - Consistent pricing approach
                """)
        
        with col2:
            st.markdown("### Business Recommendations")
            
            if loan_data['risk_diff_contract'] > 5:
                st.info("""
                **For Higher Risk Loan Types:**
                1. **Stricter Criteria:** Higher income/credit requirements
                2. **Lower Limits:** Conservative credit limits
                3. **Additional Collateral:** Require security
                4. **Higher Pricing:** Risk-adjusted interest rates
                
                **Portfolio Management:**
                1. **Diversification:** Balance across loan types
                2. **Monitoring:** Closer watch on high-risk types
                3. **Review:** Regular risk assessment by type
                4. **Limits:** Caps on high-risk type exposure
                """)
            else:
                st.info("""
                **Standard Loan Type Approach:**
                1. **Consistent Criteria:** Similar requirements across types
                2. **Customer Focus:** Let customer needs drive type selection
                3. **Risk-Based:** Use other risk factors primarily
                4. **Market Competitive:** Align with industry practices
                """)
        
        # Detailed statistics
        with st.expander("📊 Detailed Loan Type Statistics"):
            st.dataframe(
                loan_data['contract_type_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

def show_e12(preprocessed, app_data, prev_data):
    """Display E12: Loan Purpose Analysis"""
    st.header("🎯 E12: Do different loan purposes show distinct risk profiles?")
    
    if 'loan_purpose_analysis' in preprocessed:
        purpose_data = preprocessed['loan_purpose_analysis']
        
        # Top row: Purpose analysis
        st.subheader("Loan Purpose Risk Analysis")
        
        # Show top 20 purposes for better visualization
        top_purposes = purpose_data['purpose_analysis'].head(20)
        
        fig = px.bar(
            top_purposes,
            x='Loan Purpose',
            y='Default Rate (%)',
            title='Default Rate by Loan Purpose (Top 20)',
            color='Default Rate (%)',
            color_continuous_scale='Reds',
            text='Default Rate (%)',
            hover_data=['Count']
        )
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        fig.update_layout(
            height=600,
            xaxis_title="Loan Purpose",
            yaxis_title="Default Rate (%)",
            xaxis_tickangle=-45,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Middle row: Risk comparison
        st.subheader("Highest vs Lowest Risk Purposes")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Top risky purposes
            st.markdown("### 🔴 Top 3 Riskiest Purposes")
            for idx, row in purpose_data['top_risky_purposes'].head(3).iterrows():
                st.metric(
                    row['Loan Purpose'][:25] + ("..." if len(row['Loan Purpose']) > 25 else ""),
                    f"{row['Default Rate (%)']:.1f}%",
                    f"{row['Count']:,} applicants"
                )
        
        with col2:
            # Safest purposes
            st.markdown("### 🟢 Top 3 Safest Purposes")
            for idx, row in purpose_data['safest_purposes'].head(3).iterrows():
                st.metric(
                    row['Loan Purpose'][:25] + ("..." if len(row['Loan Purpose']) > 25 else ""),
                    f"{row['Default Rate (%)']:.1f}%",
                    f"{row['Count']:,} applicants"
                )
        
        with col3:
            # Overall statistics
            st.markdown("### 📊 Overall Statistics")
            
            st.metric(
                "Risk Range",
                f"{purpose_data['risk_range_purpose']:.1f} pp",
                "Highest vs Lowest"
            )
            
            avg_default_rate = purpose_data['purpose_analysis']['Default Rate (%)'].mean()
            st.metric(
                "Average Default Rate",
                f"{avg_default_rate:.1f}%",
                "Across all purposes"
            )
            
            total_applicants = purpose_data['purpose_analysis']['Count'].sum()
            st.metric(
                "Total Applications",
                f"{total_applicants:,}",
                "With purpose data"
            )
        
        # Bottom row: Risk categories and recommendations
        st.subheader("Purpose-Based Risk Categories")
        
        # Categorize purposes by risk level
        if not purpose_data['purpose_analysis'].empty:
            avg_rate = purpose_data['purpose_analysis']['Default Rate (%)'].mean()
            std_rate = purpose_data['purpose_analysis']['Default Rate (%)'].std()
            
            high_risk_threshold = avg_rate + std_rate
            low_risk_threshold = avg_rate - std_rate
            
            high_risk_purposes = purpose_data['purpose_analysis'][purpose_data['purpose_analysis']['Default Rate (%)'] > high_risk_threshold]
            medium_risk_purposes = purpose_data['purpose_analysis'][
                (purpose_data['purpose_analysis']['Default Rate (%)'] >= low_risk_threshold) &
                (purpose_data['purpose_analysis']['Default Rate (%)'] <= high_risk_threshold)
            ]
            low_risk_purposes = purpose_data['purpose_analysis'][purpose_data['purpose_analysis']['Default Rate (%)'] < low_risk_threshold]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### 🔴 High Risk ({len(high_risk_purposes)} purposes)")
                st.write(f"> {high_risk_threshold:.1f}%+ default rate")
                for idx, row in high_risk_purposes.head(5).iterrows():
                    st.write(f"- **{row['Loan Purpose'][:20]}...**: {row['Default Rate (%)']:.1f}%")
            
            with col2:
                st.markdown(f"### 🟡 Medium Risk ({len(medium_risk_purposes)} purposes)")
                st.write(f"{low_risk_threshold:.1f}% to {high_risk_threshold:.1f}%")
                for idx, row in medium_risk_purposes.head(5).iterrows():
                    st.write(f"- **{row['Loan Purpose'][:20]}...**: {row['Default Rate (%)']:.1f}%")
            
            with col3:
                st.markdown(f"### 🟢 Low Risk ({len(low_risk_purposes)} purposes)")
                st.write(f"< {low_risk_threshold:.1f}% default rate")
                for idx, row in low_risk_purposes.head(5).iterrows():
                    st.write(f"- **{row['Loan Purpose'][:20]}...**: {row['Default Rate (%)']:.1f}%")
        
        # Business recommendations
        st.subheader("Business Recommendations")
        
        if purpose_data['risk_range_purpose'] > 15:
            st.info("""
            **For High Risk Purposes:**
            1. **Additional Verification:** Require proof of purpose
            2. **Stricter Criteria:** Higher income/credit requirements
            3. **Lower Limits:** Conservative loan amounts
            4. **Monitoring:** Track purpose-specific performance
            
            **Purpose-Based Strategies:**
            1. **Risk-Based Pricing:** Higher rates for high-risk purposes
            2. **Purpose Verification:** Validate stated purpose
            3. **Segment Limits:** Caps on high-risk purpose exposure
            4. **Regular Review:** Update purpose risk assessments
            """)
        elif purpose_data['risk_range_purpose'] > 8:
            st.info("""
            **Moderate Risk Differentiation:**
            1. **Risk Weighting:** Include purpose in risk scoring
            2. **Approval Criteria:** Adjust requirements by purpose
            3. **Portfolio Balance:** Monitor purpose mix
            4. **Customer Education:** Guide towards lower-risk purposes
            
            **Recommended Actions:**
            1. **Documentation:** Require purpose documentation
            2. **Analysis:** Regular purpose performance review
            3. **Guidance:** Suggest alternative purposes if high-risk
            4. **Monitoring:** Watch for purpose misuse
            """)
        else:
            st.info("""
            **Limited Risk Differentiation:**
            1. **Standard Approach:** Consistent criteria across purposes
            2. **Customer Choice:** Let needs drive purpose selection
            3. **Focus Elsewhere:** Prioritize other risk factors
            4. **Market Alignment:** Follow industry practices
            
            **Best Practices:**
            1. **Documentation:** Maintain purpose records
            2. **Analysis:** Monitor for emerging patterns
            3. **Flexibility:** Adapt to market changes
            4. **Customer Service:** Help choose appropriate purposes
            """)
        
        # Detailed statistics
        with st.expander("📊 Detailed Purpose Statistics"):
            st.dataframe(
                purpose_data['purpose_analysis'].style.format({
                    'Default Rate (%)': '{:.2f}%',
                    'Count': '{:,}'
                }),
                use_container_width=True
            )

def show_f13(preprocessed, app_data, prev_data):
    """Display F13: Social Circle Indicators analysis"""
    st.header("👥 F13: Do social-circle default indicators add predictive value?")
    
    if 'social_circle_analysis' in preprocessed:
        social_data = preprocessed['social_circle_analysis']
        
        # Top row: 30-day and 60-day analysis
        st.subheader("Social Circle Default Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 30-day analysis
            if not social_data['social_30_analysis'].empty:
                fig = px.bar(
                    social_data['social_30_analysis'],
                    x='Social Circle 30-Day Default Rate',
                    y='Default Rate (%)',
                    title='30-Day Social Circle Default Impact',
                    color='Default Rate (%)',
                    color_continuous_scale='Reds',
                    text='Default Rate (%)',
                    hover_data=['Count']
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Social Circle 30-Day Default Rate",
                    yaxis_title="Applicant Default Rate (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("30-day social circle data not available.")
        
        with col2:
            # 60-day analysis
            if not social_data['social_60_analysis'].empty:
                fig = px.bar(
                    social_data['social_60_analysis'],
                    x='Social Circle 60-Day Default Rate',
                    y='Default Rate (%)',
                    title='60-Day Social Circle Default Impact',
                    color='Default Rate (%)',
                    color_continuous_scale='Reds',
                    text='Default Rate (%)',
                    hover_data=['Count']
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Social Circle 60-Day Default Rate",
                    yaxis_title="Applicant Default Rate (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("60-day social circle data not available.")
        
        # Middle row: Correlation analysis
        st.subheader("Social Circle Correlation Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if social_data['corr_30_days'] is not None and not np.isnan(social_data['corr_30_days']):
                st.metric(
                    "30-Day Correlation",
                    f"{social_data['corr_30_days']:.3f}",
                    "Positive = Higher social defaults → Higher risk"
                )
            
            if not social_data['social_30_analysis'].empty:
                max_30_rate = social_data['social_30_analysis']['Default Rate (%)'].max()
                min_30_rate = social_data['social_30_analysis']['Default Rate (%)'].min()
                st.metric(
                    "30-Day Risk Range",
                    f"{max_30_rate - min_30_rate:.1f} pp",
                    f"{max_30_rate:.1f}% vs {min_30_rate:.1f}%"
                )
        
        with col2:
            if social_data['corr_60_days'] is not None and not np.isnan(social_data['corr_60_days']):
                st.metric(
                    "60-Day Correlation",
                    f"{social_data['corr_60_days']:.3f}",
                    "Positive = Higher social defaults → Higher risk"
                )
            
            if not social_data['social_60_analysis'].empty:
                max_60_rate = social_data['social_60_analysis']['Default Rate (%)'].max()
                min_60_rate = social_data['social_60_analysis']['Default Rate (%)'].min()
                st.metric(
                    "60-Day Risk Range",
                    f"{max_60_rate - min_60_rate:.1f} pp",
                    f"{max_60_rate:.1f}% vs {min_60_rate:.1f}%"
                )
        
        with col3:
            # Determine which is more predictive
            if (social_data['corr_30_days'] is not None and not np.isnan(social_data['corr_30_days'])) and \
               (social_data['corr_60_days'] is not None and not np.isnan(social_data['corr_60_days'])):
                
                if abs(social_data['corr_30_days']) > abs(social_data['corr_60_days']):
                    stronger_predictor = "30-Day"
                    correlation_diff = abs(social_data['corr_30_days']) - abs(social_data['corr_60_days'])
                else:
                    stronger_predictor = "60-Day"
                    correlation_diff = abs(social_data['corr_60_days']) - abs(social_data['corr_30_days'])
                
                st.metric(
                    "Stronger Predictor",
                    stronger_predictor,
                    f"{correlation_diff:.3f} higher correlation"
                )
            
            # Overall assessment
            if social_data['corr_30_days'] is not None and not np.isnan(social_data['corr_30_days']):
                if abs(social_data['corr_30_days']) > 0.15:
                    predictive_strength = "Strong"
                elif abs(social_data['corr_30_days']) > 0.05:
                    predictive_strength = "Moderate"
                else:
                    predictive_strength = "Weak"
                
                st.metric(
                    "Predictive Strength",
                    predictive_strength,
                    "Based on correlation"
                )
        
        # Bottom row: Insights and recommendations
        st.subheader("Social Circle Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk assessment
            if social_data['corr_30_days'] is not None and abs(social_data['corr_30_days']) > 0.15:
                st.error("""
                **⚠️ Strong Social Circle Effect**  
                Social circle defaults are powerful predictors.
                
                **Key Findings:**
                - "Birds of a feather flock together"
                - Social networks influence financial behavior
                - High contagion effect in defaults
                - Strong early warning signal
                """)
            elif social_data['corr_30_days'] is not None and abs(social_data['corr_30_days']) > 0.05:
                st.warning("""
                **⚠️ Moderate Social Circle Effect**  
                Social networks show meaningful correlation.
                
                **Key Findings:**
                - Some evidence of social influence
                - Useful as supplementary factor
                - Consider regional/community factors
                - Monitor for stronger patterns
                """)
            else:
                st.success("""
                **✅ Weak Social Circle Effect**  
                Limited predictive power from social networks.
                
                **Key Findings:**
                - Individual factors more important
                - Social influence may be indirect
                - Focus on personal financial behavior
                - Consider data quality issues
                """)
        
        with col2:
            # Business implications
            st.markdown("### Business Implications")
            
            if social_data['corr_30_days'] is not None and abs(social_data['corr_30_days']) > 0.15:
                st.info("""
                **High Predictive Value Actions:**
                1. **Risk Models:** Include social circle data in scoring
                2. **Early Warning:** Monitor social network changes
                3. **Community Analysis:** Identify high-risk networks
                4. **Targeted Interventions:** Community-based programs
                
                **Ethical Considerations:**
                1. **Fair Lending:** Ensure not discriminatory
                2. **Transparency:** Explain factors used
                3. **Data Privacy:** Protect social network information
                4. **Individual Focus:** Balance with personal factors
                """)
            else:
                st.info("""
                **Standard Approach:**
                1. **Supplementary Use:** Consider as secondary factor
                2. **Pattern Recognition:** Look for extreme cases only
                3. **Combine Factors:** Use with traditional metrics
                4. **Regular Review:** Monitor changing patterns
                
                **Best Practices:**
                1. **Data Quality:** Ensure accurate social circle data
                2. **Context Analysis:** Consider community factors
                3. **Holistic View:** Balance multiple risk factors
                4. **Continuous Learning:** Update models as patterns emerge
                """)
        
        # Critical thresholds
        st.markdown("### Critical Risk Thresholds")
        
        if not social_data['social_30_analysis'].empty:
            # Find critical threshold where risk increases significantly
            social_30_data = social_data['social_30_analysis']
            
            # Look for significant jumps in default rate
            if len(social_30_data) >= 3:
                risk_increases = []
                for i in range(1, len(social_30_data)):
                    increase = social_30_data.iloc[i]['Default Rate (%)'] - social_30_data.iloc[i-1]['Default Rate (%)']
                    risk_increases.append({
                        'From': social_30_data.iloc[i-1]['Social Circle 30-Day Default Rate'],
                        'To': social_30_data.iloc[i]['Social Circle 30-Day Default Rate'],
                        'Increase': increase
                    })
                
                # Find the largest increase
                if risk_increases:
                    largest_increase = max(risk_increases, key=lambda x: x['Increase'])
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Critical Threshold",
                            f"{largest_increase['To']}",
                            f"Risk increases significantly"
                        )
                    
                    with col2:
                        st.metric(
                            "Risk Jump",
                            f"{largest_increase['Increase']:.1f} pp",
                            f"At {largest_increase['To']} threshold"
                        )
                    
                    with col3:
                        st.metric(
                            "Action Point",
                            "Monitor Closely",
                            f"Above {largest_increase['From']} social defaults"
                        )
        
        # Detailed statistics
        with st.expander("📊 Detailed Social Circle Statistics"):
            tab1, tab2 = st.tabs(["30-Day Analysis", "60-Day Analysis"])
            
            with tab1:
                if not social_data['social_30_analysis'].empty:
                    st.dataframe(
                        social_data['social_30_analysis'].style.format({
                            'Default Rate (%)': '{:.2f}%',
                            'Count': '{:,}'
                        }),
                        use_container_width=True
                    )
            
            with tab2:
                if not social_data['social_60_analysis'].empty:
                    st.dataframe(
                        social_data['social_60_analysis'].style.format({
                            'Default Rate (%)': '{:.2f}%',
                            'Count': '{:,}'
                        }),
                        use_container_width=True
                    )

def show_f14(preprocessed, app_data, prev_data):
    """Display F14: Document Submission Behavior analysis"""
    st.header("📄 F14: Does document submission behavior indicate reliability?")
    
    if 'document_analysis' in preprocessed:
        doc_data = preprocessed['document_analysis']
        
        # Top row: Documents submitted analysis
        st.subheader("Document Submission Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = px.bar(
                doc_data['documents_submitted_analysis'],
                x='Documents Submitted',
                y='Default Rate (%)',
                title='Default Rate by Number of Documents Submitted',
                color='Default Rate (%)',
                color_continuous_scale='RdYlGn_r',
                text='Default Rate (%)',
                hover_data=['Mean Documents', 'Count']
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            fig.update_layout(
                height=400,
                xaxis_title="Number of Documents Submitted",
                yaxis_title="Default Rate (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric(
                "No Documents",
                f"{doc_data['no_docs_rate']:.1f}%",
                "Highest risk"
            )
            
            st.metric(
                "Optimal Documents",
                f"{doc_data['max_docs_rate']:.1f}%",
                "Lowest default rate"
            )
            
            st.metric(
                "Correlation",
                f"{doc_data['corr_documents']:.3f}",
                "Negative = More docs → Lower risk"
            )
            
            risk_reduction = doc_data['no_docs_rate'] - doc_data['max_docs_rate']
            st.metric(
                "Risk Reduction",
                f"{risk_reduction:.1f} pp",
                "With optimal documentation"
            )
        
        # Middle row: Specific document analysis
        st.subheader("Specific Document Impact Analysis")
        
        if not doc_data['specific_docs_analysis'].empty:
            # Show top 10 documents with biggest impact
            top_docs = doc_data['specific_docs_analysis'].head(10)
            
            fig = px.bar(
                top_docs,
                x='Document Type',
                y='Default Rate Difference (%)',
                title='Impact of Specific Document Submission',
                color='Default Rate Difference (%)',
                color_continuous_scale='RdYlGn_r',
                text='Default Rate Difference (%)',
                hover_data=['Submitted Rate', 'Not Submitted Rate']
            )
            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )
            fig.update_layout(
                height=500,
                xaxis_title="Document Type",
                yaxis_title="Default Rate Difference (Submitted - Not Submitted)",
                xaxis_tickangle=-45,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Key document insights
            st.markdown("### Key Document Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Most Protective Documents")
                protective_docs = top_docs[top_docs['Default Rate Difference (%)'] < 0].sort_values('Default Rate Difference (%)').head(3)
                for idx, row in protective_docs.iterrows():
                    reduction = abs(row['Default Rate Difference (%)'])
                    st.metric(
                        row['Document Type'],
                        f"{-reduction:.1f} pp reduction",
                        f"Submitted: {row['Submitted Rate']:.1f}% vs Not: {row['Not Submitted Rate']:.1f}%"
                    )
            
            with col2:
                st.markdown("#### 📉 Risk-Increasing Documents")
                risk_docs = top_docs[top_docs['Default Rate Difference (%)'] > 0].sort_values('Default Rate Difference (%)', ascending=False).head(3)
                for idx, row in risk_docs.iterrows():
                    st.metric(
                        row['Document Type'],
                        f"+{row['Default Rate Difference (%)']:.1f} pp increase",
                        f"Submitted: {row['Submitted Rate']:.1f}% vs Not: {row['Not Submitted Rate']:.1f}%"
                    )
        
        # Bottom row: Risk analysis and recommendations
        st.subheader("Documentation Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overall assessment
            if doc_data['corr_documents'] < -0.2:
                st.error("""
                **⚠️ Strong Documentation Effect**  
                Document submission is a powerful predictor.
                
                **Key Findings:**
                - Documentation correlates strongly with reliability
                - Willingness to provide documents indicates transparency
                - Documented applicants are significantly lower risk
                - Critical factor in risk assessment
                """)
            elif doc_data['corr_documents'] < -0.1:
                st.warning("""
                **⚠️ Moderate Documentation Effect**  
                Document submission shows meaningful correlation.
                
                **Key Findings:**
                - Better documented applications perform better
                - Documentation indicates preparedness
                - Useful as risk factor
                - Consider in approval decisions
                """)
            else:
                st.success("""
                **✅ Weak Documentation Effect**  
                Limited predictive power from documentation.
                
                **Key Findings:**
                - Documentation not a strong differentiator
                - Focus on document quality over quantity
                - Other factors more important
                - Consider data completeness issues
                """)
        
        with col2:
            # Business implications
            st.markdown("### Business Implications")
            
            if doc_data['corr_documents'] < -0.15:
                st.info("""
                **For Strong Documentation Effect:**
                1. **Mandatory Documents:** Require key protective documents
                2. **Document Incentives:** Encourage complete documentation
                3. **Risk-Based Requirements:** More docs for higher risk
                4. **Automated Scoring:** Include in automated systems
                
                **Document Strategy:**
                1. **Priority Documents:** Focus on most protective docs
                2. **Streamlined Process:** Make submission easy
                3. **Clear Requirements:** Communicate document needs
                4. **Verification:** Ensure document authenticity
                """)
            else:
                st.info("""
                **Standard Documentation Approach:**
                1. **Basic Requirements:** Standard document set
                2. **Quality Focus:** Emphasize document accuracy
                3. **Risk Balance:** Don't over-rely on documentation
                4. **Customer Experience:** Simplify document process
                
                **Best Practices:**
                1. **Consistent Requirements:** Standard across applicants
                2. **Clear Communication:** Explain document purposes
                3. **Efficient Process:** Minimize customer burden
                4. **Data Integrity:** Ensure document data quality
                """)
        
        # Document optimization
        st.markdown("### Document Optimization Strategy")
        
        if not doc_data['specific_docs_analysis'].empty:
            # Identify optimal document set
            protective_docs = doc_data['specific_docs_analysis'][doc_data['specific_docs_analysis']['Default Rate Difference (%)'] < 0]
            
            if len(protective_docs) >= 3:
                top_protective = protective_docs.sort_values('Default Rate Difference (%)').head(3)
                
                col1, col2, col3 = st.columns(3)
                
                docs = top_protective['Document Type'].tolist()
                reductions = [abs(row['Default Rate Difference (%)']) for idx, row in top_protective.iterrows()]
                
                for i, (doc, reduction) in enumerate(zip(docs, reductions)):
                    with [col1, col2, col3][i]:
                        st.metric(
                            f"Key Document {i+1}",
                            doc[:15] + ("..." if len(doc) > 15 else ""),
                            f"{reduction:.1f} pp risk reduction"
                        )
                
                st.info(f"""
                **Recommended Core Document Set:** {', '.join([d[:20] + '...' if len(d) > 20 else d for d in docs])}
                
                **Strategy:** Focus on collecting these 3 documents first, as they provide the strongest risk reduction.
                """)
        
        # Detailed statistics
        with st.expander("📊 Detailed Document Statistics"):
            tab1, tab2 = st.tabs(["Documents Submitted", "Specific Document Impact"])
            
            with tab1:
                st.dataframe(
                    doc_data['documents_submitted_analysis'].style.format({
                        'Default Rate (%)': '{:.2f}%',
                        'Mean Documents': '{:.1f}',
                        'Count': '{:,}'
                    }),
                    use_container_width=True
                )
            
            with tab2:
                if not doc_data['specific_docs_analysis'].empty:
                    st.dataframe(
                        doc_data['specific_docs_analysis'].style.format({
                            'Default Rate Difference (%)': '{:.2f}%',
                            'Submitted Rate': '{:.2f}%',
                            'Not Submitted Rate': '{:.2f}%'
                        }),
                        use_container_width=True
                    )

def show_f15(preprocessed, app_data, prev_data):
    """Display F15: Region Ratings Analysis"""
    st.header("🗺️ F15: Do region ratings show clear differences in default levels?")
    
    if 'region_analysis' in preprocessed:
        region_data = preprocessed['region_analysis']
        
        # Top row: Region rating analysis
        st.subheader("Region Rating Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Client region rating
            if not region_data['region_client_analysis'].empty:
                fig = px.bar(
                    region_data['region_client_analysis'],
                    x='Region Rating',
                    y='Default Rate (%)',
                    title='Default Rate by Region Rating',
                    color='Default Rate (%)',
                    color_continuous_scale='Reds',
                    text='Default Rate (%)',
                    hover_data=['Count']
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Region Rating (1 = Best, 3 = Worst)",
                    yaxis_title="Default Rate (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Client region metrics
            if not region_data['region_client_analysis'].empty:
                worst_client = region_data['region_client_analysis'].loc[region_data['region_client_analysis']['Default Rate (%)'].idxmax()]
                best_client = region_data['region_client_analysis'].loc[region_data['region_client_analysis']['Default Rate (%)'].idxmin()]
                
                st.metric(
                    "Worst Region Rating",
                    f"Rating {worst_client['Region Rating']}",
                    f"{worst_client['Default Rate (%)']:.1f}% default rate"
                )
                
                st.metric(
                    "Best Region Rating",
                    f"Rating {best_client['Region Rating']}",
                    f"{best_client['Default Rate (%)']:.1f}% default rate"
                )
        
        with col2:
            # City region rating
            if not region_data['region_city_analysis'].empty:
                fig = px.bar(
                    region_data['region_city_analysis'],
                    x='Region Rating (with City)',
                    y='Default Rate (%)',
                    title='Default Rate by Region Rating (with City)',
                    color='Default Rate (%)',
                    color_continuous_scale='Reds',
                    text='Default Rate (%)',
                    hover_data=['Count']
                )
                fig.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                fig.update_layout(
                    height=400,
                    xaxis_title="Region Rating with City (1 = Best, 3 = Worst)",
                    yaxis_title="Default Rate (%)",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # City region metrics
            if not region_data['region_city_analysis'].empty:
                worst_city = region_data['region_city_analysis'].loc[region_data['region_city_analysis']['Default Rate (%)'].idxmax()]
                best_city = region_data['region_city_analysis'].loc[region_data['region_city_analysis']['Default Rate (%)'].idxmin()]
                
                st.metric(
                    "Worst City Region",
                    f"Rating {worst_city['Region Rating (with City)']}",
                    f"{worst_city['Default Rate (%)']:.1f}% default rate"
                )
                
                st.metric(
                    "Best City Region",
                    f"Rating {best_city['Region Rating (with City)']}",
                    f"{best_city['Default Rate (%)']:.1f}% default rate"
                )
        
        # Middle row: Correlation and trend analysis
        st.subheader("Region Rating Correlation Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if region_data['corr_client_rating'] is not None and not np.isnan(region_data['corr_client_rating']):
                st.metric(
                    "Client Rating Correlation",
                    f"{region_data['corr_client_rating']:.3f}",
                    "Positive = Higher rating (worse) → Higher risk"
                )
        
        with col2:
            if region_data['corr_city_rating'] is not None and not np.isnan(region_data['corr_city_rating']):
                st.metric(
                    "City Rating Correlation",
                    f"{region_data['corr_city_rating']:.3f}",
                    "Positive = Higher rating (worse) → Higher risk"
                )
        
        with col3:
            st.metric(
                "Risk Range",
                f"{region_data['risk_range_region']:.1f} pp",
                "Difference between best and worst ratings"
            )
        
        # Trend analysis
        if not region_data['region_client_analysis'].empty and len(region_data['region_client_analysis']) >= 3:
            st.subheader("Region Rating Trend Analysis")
            
            fig = px.line(
                region_data['region_client_analysis'],
                x='Region Rating',
                y='Default Rate (%)',
                title='Default Rate Trend Across Region Ratings',
                markers=True,
                line_shape='spline',
                text='Default Rate (%)'
            )
            fig.update_traces(
                textposition='top center',
                line=dict(width=3, color='darkred'),
                marker=dict(size=10, color='red')
            )
            fig.update_layout(
                height=400,
                xaxis_title="Region Rating (1 = Best, 3 = Worst)",
                yaxis_title="Default Rate (%)",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Bottom row: Risk analysis and geographic strategy
        st.subheader("Geographic Risk Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk assessment
            if region_data['risk_range_region'] > 10:
                st.error("""
                **⚠️ Extreme Geographic Variation**  
                Region ratings show massive risk differences.
                
                **Key Findings:**
                - Geographic location is critical risk factor
                - Regional economic conditions heavily influence defaults
                - Strong need for geographic risk differentiation
                - Consider local economic indicators
                """)
            elif region_data['risk_range_region'] > 5:
                st.warning("""
                **⚠️ Significant Geographic Variation**  
                Region ratings are strong predictors.
                
                **Key Findings:**
                - Meaningful differences across regions
                - Geographic factors important in risk assessment
                - Consider regional economic data
                - Monitor regional trends
                """)
            else:
                st.success("""
                **✅ Moderate Geographic Variation**  
                Limited differences across region ratings.
                
                **Key Findings:**
                - Geographic factors less critical
                - Focus on individual characteristics
                - Regional differences may be overstated
                - Consider other location factors
                """)
        
        with col2:
            # Geographic strategy
            st.markdown("### Geographic Strategy")
            
            if region_data['risk_range_region'] > 8:
                st.info("""
                **For High Geographic Variation:**
                1. **Regional Risk Models:** Develop location-specific models
                2. **Geographic Limits:** Set exposure limits by region
                3. **Local Underwriting:** Consider local economic conditions
                4. **Regional Monitoring:** Track performance by geography
                
                **Market Strategy:**
                1. **Targeted Marketing:** Focus on lower-risk regions
                2. **Risk-Based Pricing:** Adjust rates by region
                3. **Local Partnerships:** Work with local institutions
                4. **Economic Analysis:** Monitor regional economies
                """)
            else:
                st.info("""
                **Standard Geographic Approach:**
                1. **Consistent Criteria:** Standard across regions
                2. **National Focus:** Prioritize national factors
                3. **Portfolio Balance:** Maintain geographic diversity
                4. **Market Coverage:** Serve all regions equitably
                
                **Best Practices:**
                1. **Fair Lending:** Ensure equal access across regions
                2. **Data Collection:** Gather regional economic data
                3. **Trend Monitoring:** Watch for emerging patterns
                4. **Adaptive Strategy:** Adjust as patterns change
                """)
        
        # Geographic concentration analysis
        st.markdown("### Geographic Concentration Analysis")
        
        if not region_data['region_client_analysis'].empty:
            # Calculate portfolio concentration
            total_applicants = region_data['region_client_analysis']['Count'].sum()
            high_risk_share = region_data['region_client_analysis'][
                region_data['region_client_analysis']['Default Rate (%)'] > 
                region_data['region_client_analysis']['Default Rate (%)'].mean()
            ]['Count'].sum() / total_applicants * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "High-Risk Region Share",
                    f"{high_risk_share:.1f}%",
                    "Of portfolio in above-average risk regions"
                )
            
            with col2:
                # Worst region concentration
                worst_region_share = region_data['worst_rating_client']['Count'] / total_applicants * 100
                st.metric(
                    "Worst Region Concentration",
                    f"{worst_region_share:.1f}%",
                    f"{region_data['worst_rating_client']['Count']:,} applicants"
                )
            
            with col3:
                # Risk concentration score
                concentration_score = high_risk_share * (region_data['risk_range_region'] / 100)
                st.metric(
                    "Geographic Risk Score",
                    f"{concentration_score:.2f}",
                    "Higher = More concentrated risk"
                )
        
        # Detailed statistics
        with st.expander("📊 Detailed Region Statistics"):
            tab1, tab2 = st.tabs(["Client Region", "City Region"])
            
            with tab1:
                if not region_data['region_client_analysis'].empty:
                    st.dataframe(
                        region_data['region_client_analysis'].style.format({
                            'Default Rate (%)': '{:.2f}%',
                            'Count': '{:,}'
                        }),
                        use_container_width=True
                    )
            
            with tab2:
                if not region_data['region_city_analysis'].empty:
                    st.dataframe(
                        region_data['region_city_analysis'].style.format({
                            'Default Rate (%)': '{:.2f}%',
                            'Count': '{:,}'
                        }),
                        use_container_width=True
                    )
