import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import warnings
import pickle
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# Import modular components
from data_loader import load_data, load_preprocessed_data, load_data_for_overview
from visualization import (
    create_risk_segment_analysis,
    create_demographic_analysis,
    create_financial_analysis,
    create_age_analysis,
    create_income_vs_credit_scatter,
    create_occupation_analysis,
    create_loan_type_analysis,
    create_purpose_risk_analysis,
    create_correlation_heatmap,
    create_default_distribution_by_feature,
    create_age_credit_interaction,
    create_debt_stress_analysis,
    create_education_employment_analysis,
    create_living_standard_analysis,
    create_employment_tenure_occupation_analysis
)
from question_analytics import (
    show_overview,
    show_a1,
    show_a2,
    show_a3,
    show_b4,
    show_b5,
    show_b6,
    show_c7,
    show_c8,
    show_d9,
    show_d10,
    show_e11,
    show_e12,
    show_f13,
    show_f14,
    show_f15,
)
from utils import (
    setup_page_config,
    setup_custom_css,
    display_header,
    display_kpi_metrics,
    create_filters_sidebar,
)


# ==================== MAIN APPLICATION ====================
def main():
    """Main dashboard application"""

    # Setup page configuration
    setup_page_config()
    setup_custom_css()

    # Display header
    display_header()

    # Load all necessary data
    with st.spinner("Loading datasets..."):
        try:
            df = load_data()
            try:
                preprocessed = load_preprocessed_data()
                app_data, prev_data = load_data_for_overview()
                qa_data_available = True
            except:
                qa_data_available = False
                preprocessed = {}
                app_data, prev_data = None, None
                st.warning("⚠️ Q&A data not available. Running in analytics-only mode.")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    # Create sidebar filters
    df = create_filters_sidebar(df)

    # Display KPI metrics
    display_kpi_metrics(df)

    # Create tabs
    tabs = st.tabs(
        [
            "📊 Overview & Insights",
            "🎯 Risk Segmentation",
            "👤 Demographics",
            "💰 Financial Analysis",
            "🏦 Loan Structure",
            "📊 Employment and Education",
            "📈 Correlation Analytics",
            "❓ Q&A Analysis",
        ]
    )

    # TAB 1: Overview
    with tabs[0]:
        st.header("📊 Overview & Key Insights")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Dataset Overview")
            default_rate = f"{df['TARGET'].mean() * 100:.2f}%" if 'TARGET' in df.columns else "N/A"
            avg_age = f"{df['AGE'].mean():.1f}" if 'AGE' in df.columns else "N/A"
            avg_income = f"${df['AMT_INCOME_TOTAL'].mean():,.0f}" if 'AMT_INCOME_TOTAL' in df.columns else "N/A"

            st.info(
                f"""
            **Total Applications:** {len(df):,}  
            **Default Rate:** {default_rate}  
            **Average Age:** {avg_age}  
            **Average Income:** {avg_income}
            """
            )


        with col2:
            if qa_data_available:
                show_overview(preprocessed, app_data, prev_data, compact=True)

        st.subheader("Quick Analytics")
        col1, col2 = st.columns(2)

        with col1:
            if "RISK_SEGMENT" in df.columns:
                fig = create_risk_segment_analysis(df)
                st.plotly_chart(fig, use_container_width=True, key="overview_risk_seg")

        with col2:
            if "CREDIT_INCOME_RATIO" in df.columns:
                fig = create_financial_analysis(df)
                st.plotly_chart(fig, use_container_width=True, key="overview_financial")

        st.header("🎯 Key Insights & Recommendations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            ### 📊 Risk Insights
            - **Risk segmentation** effectively stratifies default rates
            - **Younger applicants** show significantly higher default rates
            - **High credit-to-income ratios** (>5x) correlate with 3x higher defaults
            - **Certain occupations** (Laborers, Drivers) show elevated risk
            """
            )

        with col2:
            st.markdown(
                """
            ### 🚀 Business Actions
            1. **Implement risk-based pricing** using segmentation  
            2. **Add credit/income ratio caps** for high-risk groups  
            3. **Develop targeted marketing** for low-risk segments  
            4. **Create early warning systems** for high-risk indicators  
            """
            )

    # TAB 2: Risk Segmentation
    with tabs[1]:
        st.header("🎯 Risk Segmentation Analysis")

        if "RISK_SEGMENT" in df.columns:
            risk_chart = create_risk_segment_analysis(df)
            st.plotly_chart(risk_chart, use_container_width=True, key="risk_seg_chart")

            col1, col2 = st.columns([2, 1])

            with col1:
                segment_stats = (
                    df.groupby("RISK_SEGMENT")
                    .agg(
                        {
                            "TARGET": ["count", "mean", "sum"],
                            "AMT_INCOME_TOTAL": "mean",
                            "AMT_CREDIT": "mean",
                            "CREDIT_INCOME_RATIO": "mean",
                        }
                    )
                    .round(2)
                )

                segment_stats.columns = [
                    "Count",
                    "Default_Rate",
                    "Defaults",
                    "Avg_Income",
                    "Avg_Credit",
                    "Avg_Credit_Income_Ratio",
                ]
                segment_stats["Default_Rate"] = (
                    segment_stats["Default_Rate"] * 100
                ).round(1)

                st.subheader("Risk Segment Statistics")
                st.dataframe(segment_stats, use_container_width=True)

            with col2:
                chi2, p_value, _ = safe_chi2_test(df, "RISK_SEGMENT")
                st.metric(
                    "Chi-square p-value",
                    f"{p_value:.6f}",
                    delta="Significant" if p_value < 0.05 else "Not Significant",
                )

                if not segment_stats.empty:
                    highest_risk = segment_stats["Default_Rate"].idxmax()
                    highest_rate = segment_stats["Default_Rate"].max()
                    st.info(
                        f"**Key Insight:** {highest_risk} segment has the highest default rate at {highest_rate:.1f}%"
                    )

    # TAB 3: Demographics
    with tabs[2]:
        st.header("👤 Demographic Analysis")

        col1, col2 = st.columns([3, 1])

        with col1:
            if "DEMOGRAPHIC_SEGMENT" in df.columns:
                demo_chart = create_demographic_analysis(df)
                st.plotly_chart(
                    demo_chart, use_container_width=True, key="demo_analysis"
                )

            if "AGE" in df.columns:
                age_chart = create_age_analysis(df)
                st.plotly_chart(
                    age_chart, use_container_width=True, key="age_analysis_chart"
                )

        with col2:
            if "CODE_GENDER" in df.columns:
                gender_stats = (
                    df.groupby("CODE_GENDER")
                    .agg(
                        {
                            "TARGET": ["count", "mean"],
                            "AMT_INCOME_TOTAL": "mean",
                            "AGE": "mean",
                        }
                    )
                    .round(2)
                )

                gender_stats.columns = [
                    "Count",
                    "Default_Rate",
                    "Avg_Income",
                    "Avg_Age",
                ]
                gender_stats["Default_Rate"] = (
                    gender_stats["Default_Rate"] * 100
                ).round(1)

                st.subheader("Gender Analysis")
                st.dataframe(gender_stats, use_container_width=True)

        st.subheader("📊 Age vs Credit Burden Interaction")

        age_credit_fig = create_age_credit_interaction(df)
        st.plotly_chart(
            age_credit_fig,
            use_container_width=True,
            key="age_credit_interaction_tab"
        )

        st.divider()

    # TAB 4: Financial Analysis
    with tabs[3]:
        st.header("💰 Financial Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            if "CREDIT_INCOME_RATIO" in df.columns:
                financial_chart = create_financial_analysis(df)
                st.plotly_chart(
                    financial_chart, use_container_width=True, key="fin_chart"
                )

            if "AMT_INCOME_TOTAL" in df.columns and "AMT_CREDIT" in df.columns:
                scatter_chart = create_income_vs_credit_scatter(df)
                st.plotly_chart(
                    scatter_chart, use_container_width=True, key="income_credit_scatter"
                )

        with col2:
            financial_metrics = []
            if "CREDIT_INCOME_RATIO" in df.columns:
                financial_metrics.append(
                    ("Credit/Income Ratio", f"{df['CREDIT_INCOME_RATIO'].mean():.2f}")
                )
            if "AMT_INCOME_TOTAL" in df.columns:
                financial_metrics.append(
                    ("Average Income", f"${df['AMT_INCOME_TOTAL'].mean():,.0f}")
                )
                financial_metrics.append(
                    ("Income Std Dev", f"${df['AMT_INCOME_TOTAL'].std():,.0f}")
                )
            if "AMT_CREDIT" in df.columns:
                financial_metrics.append(
                    ("Average Credit", f"${df['AMT_CREDIT'].mean():,.0f}")
                )
                financial_metrics.append(
                    ("Credit Std Dev", f"${df['AMT_CREDIT'].std():,.0f}")
                )

            st.subheader("Financial Statistics")
            for metric, value in financial_metrics:
                st.metric(metric, value)

            if "CREDIT_INCOME_RATIO" in df.columns:
                ks_stat, ks_p = safe_ks_test(df, "CREDIT_INCOME_RATIO")
                st.metric(
                    "KS Test p-value",
                    f"{ks_p:.6f}",
                    delta="Significant" if ks_p < 0.05 else "Not Significant",
                )

    # TAB 5: Loan Structure
    with tabs[4]:
        st.header("🏦 Loan Structure Analysis")

        if "NAME_CONTRACT_TYPE" in df.columns:
            st.subheader("Loan Type Analysis")
            loan_chart = create_loan_type_analysis(df)
            st.plotly_chart(loan_chart, use_container_width=True, key="loan_type_chart")

        if "NAME_CASH_LOAN_PURPOSE" in df.columns:
            st.subheader("Loan Purpose Analysis")
            purpose_chart = create_purpose_risk_analysis(df)
            st.plotly_chart(
                purpose_chart, use_container_width=True, key="loan_purpose_risk"
            )


    # TAB 6: Engineered Risk Insights
    with tabs[5]:
        st.header("🧠 Employment and Education Insights")

        st.caption(
            "Feature-engineered analyses combining socio-economic, "
            "financial stress, and employment stability indicators to uncover "
            "hidden default risk patterns."
        )

        # -------------------------------------------------------
        # 1. Living Standards Analysis
        # -------------------------------------------------------
        st.subheader("🏠 Living Standards & Financial Stress")

        living_fig = create_living_standard_analysis(df)
        st.plotly_chart(
            living_fig,
            use_container_width=True,
            key="living_standard_tab"
        )

        st.divider()

        # -------------------------------------------------------
        # 2. Education + Employment Stability
        # -------------------------------------------------------
        st.subheader("🎓 Education & Employment Stability")

        edu_emp_fig = create_education_employment_analysis(df)
        st.plotly_chart(
            edu_emp_fig,
            use_container_width=True,
            key="education_employment_tab"
        )

        st.divider()

        # -------------------------------------------------------
        # 3. Debt Stress Index
        # -------------------------------------------------------
        st.subheader("💸 Debt Stress Index")

        st.caption(
            "Debt stress measures how much of a customer's income is consumed "
            "by loan obligations. Higher stress indicates lower financial resilience."
        )

        debt_stress_fig = create_debt_stress_analysis(df)
        st.plotly_chart(
            debt_stress_fig,
            use_container_width=True,
            key="debt_stress_tab"
        )

        st.divider()

        if "OCCUPATION_TYPE" in df.columns:
            st.subheader("Occupation Analysis")
            occ_chart = create_occupation_analysis(df)
            st.plotly_chart(
                occ_chart, use_container_width=True, key="occupation_analysis"
            )
        
        # -------------------------------------------------------
        # 5. Employment Tenure × Occupation
        # -------------------------------------------------------
        st.subheader("🧑‍💼 Employment Tenure & Occupation Risk")

        st.caption(
            "Analyzes how employment duration interacts with occupation type "
            "to influence default risk and income stability."
        )

        tenure_occ_fig = create_employment_tenure_occupation_analysis(df)
        st.plotly_chart(
            tenure_occ_fig,
            use_container_width=True,
            key="tenure_occupation_tab"
        )


    # TAB 7: Advanced Analytics
    with tabs[6]:
        st.header("📈 Correlation Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Variable Correlations")
            heatmap = create_correlation_heatmap(df)
            st.plotly_chart(heatmap, use_container_width=True, key="corr_heatmap_chart")

        with col2:
            st.subheader("Statistical Tests")
            test_results = []

            # if "CREDIT_INCOME_RATIO" in df.columns:
            #     ks_stat, ks_p = safe_ks_test(df, "CREDIT_INCOME_RATIO")
            #     test_results.append(
            #         {
            #             "Variable": "Credit/Income Ratio",
            #             "Test": "Kolmogorov-Smirnov",
            #             "p-value": ks_p,
            #             "Significant": ks_p < 0.05 if not np.isnan(ks_p) else False,
            #         }
            #     )

            cat_vars = ["CODE_GENDER", "NAME_CONTRACT_TYPE"]
            for var in cat_vars:
                if var in df.columns:
                    chi2, p_value, _ = safe_chi2_test(df, var)
                    test_results.append(
                        {
                            "Variable": var,
                            "Test": "Chi-square",
                            "p-value": p_value,
                            "Significant": (
                                p_value < 0.05 if not np.isnan(p_value) else False
                            ),
                        }
                    )

            if test_results:
                test_df = pd.DataFrame(test_results)
                test_df["p-value"] = test_df["p-value"].apply(
                    lambda x: f"{x:.6f}" if not np.isnan(x) else "N/A"
                )
                test_df["Significant"] = test_df["Significant"].apply(
                    lambda x: "✓" if x else "✗"
                )
                st.dataframe(test_df, use_container_width=True)

        st.subheader("Feature Distribution Analysis")

        feature_select = st.selectbox(
            "Select feature to analyze distribution:",
            options=[
                col
                for col in [
                    "AGE",
                    "AMT_INCOME_TOTAL",
                    "AMT_CREDIT",
                    "CREDIT_INCOME_RATIO",
                ]
                if col in df.columns
            ],
        )

        if feature_select:
            dist_chart = create_default_distribution_by_feature(df, feature_select)
            st.plotly_chart(
                dist_chart, use_container_width=True, key=f"dist_chart_{feature_select}"
            )

    # TAB 8: Q&A Analysis
    with tabs[7]:
        st.header("❓ Detailed Q&A Analysis")

        if not qa_data_available:
            st.error(
                "Q&A data is not available. Please ensure preprocessed_data.pkl exists."
            )
            return

        questions = [
            "👤 A1: Age Groups & Default",
            "👫 A2: Gender & Repayment",
            "👨‍👩‍👧‍👦 A3: Marital Status & Dependents",
            "💰 B4: Income Level & Default",
            "💳 B5: Credit Amount & Terms",
            "📊 B6: Income & Credit Ratios",
            "💼 C7: Employment Tenure Impact",
            "👔 C8: Employment Types & Occupations",
            "📋 D9: Previous Loan Outcomes",
            "🔍 D10: Credit Bureau Enquiries",
            "💸 E11: Loan Types (Cash vs Revolving)",
            "🎯 E12: Loan Purpose Analysis",
            "👥 F13: Social Circle Indicators",
            "📄 F14: Document Submission Behavior",
            "🗺️ F15: Region Ratings Analysis"
        ]

        col1, col2 = st.columns([1, 4])

        with col1:
            st.subheader("Questions")
            selected_question = st.radio(
                "Select a question:", questions, label_visibility="collapsed"
            )

        with col2:
            st.subheader(selected_question)

            if selected_question == "👤 A1: Age Groups & Default":
                show_a1(preprocessed, app_data, prev_data)
            elif selected_question == "👫 A2: Gender & Repayment":
                show_a2(preprocessed, app_data, prev_data)
            elif selected_question == "👨‍👩‍👧‍👦 A3: Marital Status & Dependents":
                show_a3(preprocessed, app_data, prev_data)
            elif selected_question == "💰 B4: Income Level & Default":
                show_b4(preprocessed, app_data, prev_data)
            elif selected_question == "💳 B5: Credit Amount & Terms":
                show_b5(preprocessed, app_data, prev_data)
            elif selected_question == "📊 B6: Income & Credit Ratios":
                show_b6(preprocessed, app_data, prev_data)
            elif selected_question == "💼 C7: Employment Tenure Impact":
                show_c7(preprocessed, app_data, prev_data)
            elif selected_question == "👔 C8: Employment Types & Occupations":
                show_c8(preprocessed, app_data, prev_data)
            elif selected_question == "📋 D9: Previous Loan Outcomes":
                show_d9(preprocessed, app_data, prev_data)
            elif selected_question == "🔍 D10: Credit Bureau Enquiries":
                show_d10(preprocessed, app_data, prev_data)
            elif selected_question == "💸 E11: Loan Types (Cash vs Revolving)":
                show_e11(preprocessed, app_data, prev_data)
            elif selected_question == "🎯 E12: Loan Purpose Analysis":
                show_e12(preprocessed, app_data, prev_data)
            elif selected_question == "👥 F13: Social Circle Indicators":
                show_f13(preprocessed, app_data, prev_data)
            elif selected_question == "📄 F14: Document Submission Behavior":
                show_f14(preprocessed, app_data, prev_data)
            elif selected_question == "🗺️ F15: Region Ratings Analysis":
                show_f15(preprocessed, app_data, prev_data)

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style='text-align: center; color: #6B7280;'>
        <p>📊 Loan Risk Analytics Dashboard | Built with Streamlit | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ==================== STATISTICAL FUNCTIONS ====================
def safe_chi2_test(df, column, target="TARGET"):
    """Safe chi-square test"""
    try:
        col_data = df[column].astype(str).fillna("Missing")
        ct = pd.crosstab(col_data, df[target])

        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan, np.nan, ct

        chi2, p, dof, expected = stats.chi2_contingency(ct)
        return chi2, p, ct
    except:
        return np.nan, np.nan, pd.DataFrame()


def safe_ks_test(df, column, target="TARGET"):
    """Safe KS test"""
    try:
        valid = df[[column, target]].dropna()
        if len(valid) < 20:
            return np.nan, np.nan

        g0 = valid[valid[target] == 0][column]
        g1 = valid[valid[target] == 1][column]

        if len(g0) < 5 or len(g1) < 5:
            return np.nan, np.nan

        stat, p = stats.ks_2samp(g0, g1)
        return stat, p
    except:
        return np.nan, np.nan


if __name__ == "__main__":
    main()
