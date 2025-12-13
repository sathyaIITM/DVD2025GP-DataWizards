import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

print("Loading data...")
app_data = pd.read_csv('application_data.csv')
prev_data = pd.read_csv('previous_application.csv')

preprocessed_data = {}

print("Processing Overview...")
preprocessed_data['overview'] = {
    'total_applications': len(app_data),
    'default_rate': (app_data['TARGET'].sum() / len(app_data)) * 100,
    'total_prev_applications': len(prev_data),
    'app_data_head': app_data.head(5),
    'prev_data_head': prev_data.head(5),
    'app_data_columns': len(app_data.columns),
    'prev_data_columns': len(prev_data.columns)
}

# ============================================================
# A1: Age Groups & Default
# ============================================================
print("Processing A1: Age Groups & Default...")
age_data = app_data[['TARGET', 'DAYS_BIRTH']].copy()
age_data = age_data[age_data['DAYS_BIRTH'].notna()].copy()

# Convert days to years (days are negative relative to application date)
age_data['AGE_YEARS'] = -age_data['DAYS_BIRTH'] / 365.25

# Create age groups
age_data['AGE_GROUP'] = pd.cut(
    age_data['AGE_YEARS'],
    bins=[18, 25, 30, 40, 50, 60, 100],
    labels=['18-25', '26-30', '31-40', '41-50', '51-60', '60+']
)

age_group_analysis = age_data.groupby('AGE_GROUP')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_age', lambda x: age_data.loc[x.index, 'AGE_YEARS'].mean())
]).reset_index()
age_group_analysis.columns = ['Age Group', 'Default Rate (%)', 'Count', 'Mean Age']

# Age bins for continuous analysis
age_bins = pd.cut(age_data['AGE_YEARS'], bins=10)
age_bin_analysis = age_data.groupby(age_bins)['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_age', lambda x: age_data.loc[x.index, 'AGE_YEARS'].mean())
]).reset_index()
age_bin_analysis.columns = ['Age Range', 'Default Rate (%)', 'Count', 'Mean Age']
age_bin_analysis = age_bin_analysis[age_bin_analysis['Count'] >= 100]

corr_age = np.corrcoef(age_data['AGE_YEARS'], age_data['TARGET'])[0, 1]

preprocessed_data['age_analysis'] = {
    'age_group_analysis': age_group_analysis,
    'age_bin_analysis': age_bin_analysis,
    'corr_age': corr_age,
    'youngest_group_rate': age_group_analysis.iloc[0]['Default Rate (%)'],
    'oldest_group_rate': age_group_analysis.iloc[-1]['Default Rate (%)'],
    'middle_group_rate': age_group_analysis.iloc[2]['Default Rate (%)'] if len(age_group_analysis) > 2 else np.nan,
    'risk_diff_age': age_group_analysis.iloc[0]['Default Rate (%)'] - age_group_analysis.iloc[-1]['Default Rate (%)']
}

# ============================================================
# A2: Gender & Repayment
# ============================================================
print("Processing A2: Gender & Repayment...")
gender_data = app_data[['TARGET', 'CODE_GENDER']].copy()
gender_data = gender_data[gender_data['CODE_GENDER'].isin(['M', 'F'])]

gender_default_analysis = gender_data.groupby('CODE_GENDER')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('defaults', 'sum')
]).reset_index()
gender_default_analysis.columns = ['Gender', 'Default Rate (%)', 'Count', 'Defaults']
gender_default_analysis['Gender'] = gender_default_analysis['Gender'].map({'M': 'Male', 'F': 'Female'})

gender_distribution = gender_data['CODE_GENDER'].value_counts().reset_index()
gender_distribution.columns = ['Gender', 'Count']
gender_distribution['Gender'] = gender_distribution['Gender'].map({'M': 'Male', 'F': 'Female'})

higher_risk_gender = gender_default_analysis.loc[gender_default_analysis['Default Rate (%)'].idxmax()]
lower_risk_gender = gender_default_analysis.loc[gender_default_analysis['Default Rate (%)'].idxmin()]

preprocessed_data['gender_analysis'] = {
    'gender_default_analysis': gender_default_analysis,
    'gender_distribution': gender_distribution,
    'higher_risk_gender': higher_risk_gender,
    'lower_risk_gender': lower_risk_gender,
    'risk_diff_gender': abs(gender_default_analysis.iloc[0]['Default Rate (%)'] - gender_default_analysis.iloc[1]['Default Rate (%)']) if len(gender_default_analysis) > 1 else 0
}

# ============================================================
# A3: Marital Status & Dependents
# ============================================================
print("Processing A3: Marital Status & Dependents...")
family_data = app_data[['TARGET', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN']].copy()

# Marital status analysis
marital_status_analysis = family_data.groupby('NAME_FAMILY_STATUS')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
marital_status_analysis.columns = ['Marital Status', 'Default Rate (%)', 'Count']
marital_status_analysis = marital_status_analysis.sort_values('Default Rate (%)', ascending=False)

# Dependents analysis (number of children)
dependents_data = family_data.copy()
dependents_data['CNT_CHILDREN_GROUP'] = pd.cut(
    dependents_data['CNT_CHILDREN'],
    bins=[-1, 0, 1, 2, 3, 20],
    labels=['0', '1', '2', '3', '4+']
)

dependents_analysis = dependents_data.groupby('CNT_CHILDREN_GROUP')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
dependents_analysis.columns = ['Number of Children', 'Default Rate (%)', 'Count']

corr_children = np.corrcoef(family_data['CNT_CHILDREN'], family_data['TARGET'])[0, 1]

highest_risk_marital = marital_status_analysis.iloc[0]
lowest_risk_marital = marital_status_analysis.iloc[-1]

optimal_children_row = dependents_analysis.loc[dependents_analysis['Default Rate (%)'].idxmin()]
optimal_children = optimal_children_row['Number of Children']

preprocessed_data['family_analysis'] = {
    'marital_status_analysis': marital_status_analysis,
    'dependents_analysis': dependents_analysis,
    'corr_children': corr_children,
    'highest_risk_marital': highest_risk_marital,
    'lowest_risk_marital': lowest_risk_marital,
    'risk_range_marital': highest_risk_marital['Default Rate (%)'] - lowest_risk_marital['Default Rate (%)'],
    'optimal_children': optimal_children
}

# ============================================================
# B4: Income Level & Default
# ============================================================
print("Processing B4: Income Level & Default...")
income_data = app_data[['TARGET', 'AMT_INCOME_TOTAL']].copy()
income_data = income_data[income_data['AMT_INCOME_TOTAL'] > 0].copy()

# Income quintiles
income_data['INCOME_QUINTILE'] = pd.qcut(
    income_data['AMT_INCOME_TOTAL'], 
    q=5, 
    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'],
    duplicates='drop'
)

income_quintile_analysis = income_data.groupby('INCOME_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_income', lambda x: income_data.loc[x.index, 'AMT_INCOME_TOTAL'].mean())
]).reset_index()
income_quintile_analysis.columns = ['Income Quintile', 'Default Rate (%)', 'Count', 'Mean Income']

# Income categories
income_data['INCOME_CATEGORY'] = pd.cut(
    income_data['AMT_INCOME_TOTAL'],
    bins=[0, 50000, 100000, 150000, 200000, 500000, float('inf')],
    labels=['<50k', '50k-100k', '100k-150k', '150k-200k', '200k-500k', '500k+']
)

income_category_analysis = income_data.groupby('INCOME_CATEGORY')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_income', lambda x: income_data.loc[x.index, 'AMT_INCOME_TOTAL'].mean())
]).reset_index()
income_category_analysis.columns = ['Income Category', 'Default Rate (%)', 'Count', 'Mean Income']

corr_income = np.corrcoef(income_data['AMT_INCOME_TOTAL'], income_data['TARGET'])[0, 1]

lowest_income_rate = income_quintile_analysis.iloc[0]['Default Rate (%)']
highest_income_rate = income_quintile_analysis.iloc[-1]['Default Rate (%)']

preprocessed_data['income_level_analysis'] = {
    'income_quintile_analysis': income_quintile_analysis,
    'income_category_analysis': income_category_analysis,
    'corr_income': corr_income,
    'lowest_income_rate': lowest_income_rate,
    'highest_income_rate': highest_income_rate,
    'income_risk_diff': lowest_income_rate - highest_income_rate
}

# ============================================================
# B5: Credit Amount & Terms
# ============================================================
print("Processing B5: Credit Amount & Terms...")
credit_data = app_data[['TARGET', 'AMT_CREDIT', 'AMT_ANNUITY']].copy()
credit_data = credit_data[(credit_data['AMT_CREDIT'] > 0) & (credit_data['AMT_ANNUITY'].notna())].copy()

# Credit amount analysis
credit_data['CREDIT_QUINTILE'] = pd.qcut(
    credit_data['AMT_CREDIT'], 
    q=5,
    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'],
    duplicates='drop'
)

credit_amount_analysis = credit_data.groupby('CREDIT_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_credit', lambda x: credit_data.loc[x.index, 'AMT_CREDIT'].mean())
]).reset_index()
credit_amount_analysis.columns = ['Credit Quintile', 'Default Rate (%)', 'Count', 'Mean Credit']

# Annuity analysis
credit_data['ANNUITY_QUINTILE'] = pd.qcut(
    credit_data['AMT_ANNUITY'], 
    q=5,
    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'],
    duplicates='drop'
)

annuity_analysis = credit_data.groupby('ANNUITY_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_annuity', lambda x: credit_data.loc[x.index, 'AMT_ANNUITY'].mean())
]).reset_index()
annuity_analysis.columns = ['Annuity Quintile', 'Default Rate (%)', 'Count', 'Mean Annuity']

# Loan term approximation (from previous application)
if 'CNT_PAYMENT' in prev_data.columns:
    prev_term_data = prev_data[['SK_ID_CURR', 'CNT_PAYMENT']].copy()
    prev_term_data = prev_term_data[prev_term_data['CNT_PAYMENT'].notna()].copy()
    
    # Take the most recent previous application term per client
    prev_term_data = prev_term_data.sort_values('CNT_PAYMENT', ascending=False).drop_duplicates('SK_ID_CURR')
    
    term_merged = app_data[['SK_ID_CURR', 'TARGET']].merge(prev_term_data, on='SK_ID_CURR', how='inner')
    
    if not term_merged.empty:
        term_merged['TERM_GROUP'] = pd.cut(
            term_merged['CNT_PAYMENT'],
            bins=[0, 12, 24, 36, 48, 60, float('inf')],
            labels=['<1 year', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5+ years']
        )
        
        loan_term_analysis = term_merged.groupby('TERM_GROUP')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count'),
            ('mean_term', lambda x: term_merged.loc[x.index, 'CNT_PAYMENT'].mean())
        ]).reset_index()
        loan_term_analysis.columns = ['Loan Term', 'Default Rate (%)', 'Count', 'Mean Term']
        
        corr_loan_term = np.corrcoef(term_merged['CNT_PAYMENT'], term_merged['TARGET'])[0, 1] if len(term_merged) > 1 else np.nan
    else:
        loan_term_analysis = pd.DataFrame()
        corr_loan_term = np.nan
else:
    loan_term_analysis = pd.DataFrame()
    corr_loan_term = np.nan

corr_credit_amount = np.corrcoef(credit_data['AMT_CREDIT'], credit_data['TARGET'])[0, 1]
corr_annuity = np.corrcoef(credit_data['AMT_ANNUITY'], credit_data['TARGET'])[0, 1]

preprocessed_data['credit_terms_analysis'] = {
    'credit_amount_analysis': credit_amount_analysis,
    'annuity_analysis': annuity_analysis,
    'loan_term_analysis': loan_term_analysis,
    'corr_credit_amount': corr_credit_amount,
    'corr_annuity': corr_annuity,
    'corr_loan_term': corr_loan_term
}

# ============================================================
# B6: Income & Credit Ratios
# ============================================================
print("Processing B6: Income & Credit Ratios...")
ratio_data = app_data[['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']].copy()
ratio_data = ratio_data[(ratio_data['AMT_INCOME_TOTAL'] > 0) & (ratio_data['AMT_CREDIT'] > 0) & (ratio_data['AMT_ANNUITY'].notna())].copy()

ratio_data['CREDIT_TO_INCOME_RATIO'] = ratio_data['AMT_CREDIT'] / ratio_data['AMT_INCOME_TOTAL']
ratio_data['ANNUITY_TO_INCOME_RATIO'] = ratio_data['AMT_ANNUITY'] / ratio_data['AMT_INCOME_TOTAL']

ratio_data['CREDIT_INCOME_QUINTILE'] = pd.qcut(ratio_data['CREDIT_TO_INCOME_RATIO'], q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'], duplicates='drop')
ratio_data['ANNUITY_INCOME_QUINTILE'] = pd.qcut(ratio_data['ANNUITY_TO_INCOME_RATIO'], q=5, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4', 'Q5 (Highest)'], duplicates='drop')

credit_income_analysis = ratio_data.groupby('CREDIT_INCOME_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_ratio', lambda x: ratio_data.loc[x.index, 'CREDIT_TO_INCOME_RATIO'].mean())
]).reset_index()
credit_income_analysis.columns = ['Quintile', 'Default Rate (%)', 'Count', 'Mean Ratio']

annuity_income_analysis = ratio_data.groupby('ANNUITY_INCOME_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_ratio', lambda x: ratio_data.loc[x.index, 'ANNUITY_TO_INCOME_RATIO'].mean())
]).reset_index()
annuity_income_analysis.columns = ['Quintile', 'Default Rate (%)', 'Count', 'Mean Ratio']

corr_credit_income = np.corrcoef(ratio_data['CREDIT_TO_INCOME_RATIO'], ratio_data['TARGET'])[0, 1]
corr_annuity_income = np.corrcoef(ratio_data['ANNUITY_TO_INCOME_RATIO'], ratio_data['TARGET'])[0, 1]

high_credit_income = credit_income_analysis[credit_income_analysis['Quintile'] == 'Q5 (Highest)']['Default Rate (%)'].values[0]
low_credit_income = credit_income_analysis[credit_income_analysis['Quintile'] == 'Q1 (Lowest)']['Default Rate (%)'].values[0]
high_annuity_income = annuity_income_analysis[annuity_income_analysis['Quintile'] == 'Q5 (Highest)']['Default Rate (%)'].values[0]
low_annuity_income = annuity_income_analysis[annuity_income_analysis['Quintile'] == 'Q1 (Lowest)']['Default Rate (%)'].values[0]

preprocessed_data['q1'] = {
    'credit_income_analysis': credit_income_analysis,
    'annuity_income_analysis': annuity_income_analysis,
    'corr_credit_income': corr_credit_income,
    'corr_annuity_income': corr_annuity_income,
    'high_credit_income': high_credit_income,
    'low_credit_income': low_credit_income,
    'high_annuity_income': high_annuity_income,
    'low_annuity_income': low_annuity_income
}

# ============================================================
# C7: Employment Tenure Impact
# ============================================================
print("Processing C7: Employment Tenure Impact...")
tenure_data = app_data[['TARGET', 'DAYS_EMPLOYED']].copy()
tenure_data = tenure_data[tenure_data['DAYS_EMPLOYED'].notna()].copy()

# Convert days to years (days are negative, with 365243 indicating unemployed)
tenure_data['YEARS_EMPLOYED'] = -tenure_data['DAYS_EMPLOYED'] / 365.25
# Clip unrealistic values
tenure_data['YEARS_EMPLOYED'] = tenure_data['YEARS_EMPLOYED'].clip(lower=0, upper=50)

tenure_data['TENURE_CATEGORY'] = pd.cut(
    tenure_data['YEARS_EMPLOYED'],
    bins=[-1, 0.5, 2, 5, 10, 50],
    labels=['< 6 months', '6 months - 2 years', '2-5 years', '5-10 years', '10+ years']
)

tenure_category_analysis = tenure_data.groupby('TENURE_CATEGORY')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_years', lambda x: tenure_data.loc[x.index, 'YEARS_EMPLOYED'].mean())
]).reset_index()
tenure_category_analysis.columns = ['Tenure Category', 'Default Rate (%)', 'Count', 'Mean Years']

tenure_data['TENURE_QUINTILE'] = pd.qcut(tenure_data['YEARS_EMPLOYED'], q=5, labels=['Q1 (Shortest)', 'Q2', 'Q3', 'Q4', 'Q5 (Longest)'], duplicates='drop')
tenure_quintile_analysis = tenure_data.groupby('TENURE_QUINTILE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_years', lambda x: tenure_data.loc[x.index, 'YEARS_EMPLOYED'].mean())
]).reset_index()
tenure_quintile_analysis.columns = ['Quintile', 'Default Rate (%)', 'Count', 'Mean Years']

corr_tenure = np.corrcoef(tenure_data['YEARS_EMPLOYED'], tenure_data['TARGET'])[0, 1]

shortest_tenure = tenure_category_analysis.iloc[0]['Default Rate (%)']
longest_tenure = tenure_category_analysis.iloc[-1]['Default Rate (%)']
diff_tenure = shortest_tenure - longest_tenure

preprocessed_data['q2'] = {
    'tenure_category_analysis': tenure_category_analysis,
    'tenure_quintile_analysis': tenure_quintile_analysis,
    'corr_tenure': corr_tenure,
    'shortest_tenure': shortest_tenure,
    'longest_tenure': longest_tenure,
    'diff_tenure': diff_tenure
}

# ============================================================
# C8: Employment Types & Occupations
# ============================================================
print("Processing C8: Employment Types & Occupations...")
employment_data = app_data[['TARGET', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE']].copy()

income_type_analysis = employment_data.groupby('NAME_INCOME_TYPE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('defaults', 'sum')
]).reset_index()
income_type_analysis.columns = ['Income Type', 'Default Rate (%)', 'Count', 'Defaults']
income_type_analysis = income_type_analysis.sort_values('Default Rate (%)', ascending=False)

occupation_analysis = employment_data[employment_data['OCCUPATION_TYPE'].notna()].groupby('OCCUPATION_TYPE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('defaults', 'sum')
]).reset_index()
occupation_analysis.columns = ['Occupation Type', 'Default Rate (%)', 'Count', 'Defaults']
occupation_analysis = occupation_analysis.sort_values('Default Rate (%)', ascending=False)
occupation_analysis = occupation_analysis[occupation_analysis['Count'] >= 100]

top_risk_income = income_type_analysis.head(3)
top_risk_occ = occupation_analysis.head(3)

if len(income_type_analysis) >= 2:
    highest_income = income_type_analysis.iloc[0]
    lowest_income = income_type_analysis.iloc[-1]
    risk_diff_income = highest_income['Default Rate (%)'] - lowest_income['Default Rate (%)']
    risk_range_income = risk_diff_income
else:
    highest_income = None
    lowest_income = None
    risk_diff_income = None
    risk_range_income = None

if len(occupation_analysis) >= 2:
    highest_occ = occupation_analysis.iloc[0]
    lowest_occ = occupation_analysis.iloc[-1]
    risk_diff_occ = highest_occ['Default Rate (%)'] - lowest_occ['Default Rate (%)']
else:
    highest_occ = None
    lowest_occ = None
    risk_diff_occ = None

preprocessed_data['q3'] = {
    'income_type_analysis': income_type_analysis,
    'occupation_analysis': occupation_analysis,
    'top_risk_income': top_risk_income,
    'top_risk_occ': top_risk_occ,
    'highest_income': highest_income,
    'lowest_income': lowest_income,
    'highest_occ': highest_occ,
    'lowest_occ': lowest_occ,
    'risk_diff_income': risk_diff_income,
    'risk_diff_occ': risk_diff_occ,
    'risk_range_income': risk_range_income
}

# ============================================================
# D9: Previous Loan Outcomes
# ============================================================
print("Processing D9: Previous Loan Outcomes...")
prev_status_data = prev_data[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].copy()

prev_status_summary = prev_status_data.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].agg([
    ('has_approved', lambda x: (x == 'Approved').any()),
    ('has_refused', lambda x: (x == 'Refused').any()),
    ('has_canceled', lambda x: (x == 'Canceled').any()),
    ('has_unused', lambda x: (x == 'Unused offer').any()),
    ('total_prev', 'count')
]).reset_index()

merged_prev = app_data[['SK_ID_CURR', 'TARGET']].merge(prev_status_summary, on='SK_ID_CURR', how='left')
merged_prev['has_approved'] = merged_prev['has_approved'].fillna(False)
merged_prev['has_refused'] = merged_prev['has_refused'].fillna(False)
merged_prev['has_canceled'] = merged_prev['has_canceled'].fillna(False)
merged_prev['has_prev_application'] = merged_prev['total_prev'].notna()

status_analysis = merged_prev.groupby('has_prev_application')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
status_analysis['Category'] = status_analysis['has_prev_application'].map({
    False: 'No Previous Applications',
    True: 'Has Previous Applications'
})

approved_analysis = merged_prev[merged_prev['has_prev_application']].groupby('has_approved')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
approved_analysis['Category'] = approved_analysis['has_approved'].map({
    False: 'Never Approved',
    True: 'Previously Approved'
})

outcome_analysis_data = merged_prev[merged_prev['has_prev_application']].copy()

outcome_categories = []
for idx, row in outcome_analysis_data.iterrows():
    if row['has_approved'] and not row['has_refused'] and not row['has_canceled']:
        outcome_categories.append('Only Approved')
    elif row['has_refused'] and not row['has_approved'] and not row['has_canceled']:
        outcome_categories.append('Only Refused')
    elif row['has_canceled'] and not row['has_approved'] and not row['has_refused']:
        outcome_categories.append('Only Canceled')
    elif row['has_approved'] and row['has_refused']:
        outcome_categories.append('Approved & Refused')
    elif row['has_approved'] and row['has_canceled']:
        outcome_categories.append('Approved & Canceled')
    elif row['has_refused'] and row['has_canceled']:
        outcome_categories.append('Refused & Canceled')
    elif row['has_approved'] or row['has_refused'] or row['has_canceled']:
        outcome_categories.append('Mixed Outcomes')
    else:
        outcome_categories.append('Other')

outcome_analysis_data['OUTCOME_CATEGORY'] = outcome_categories

outcome_analysis = outcome_analysis_data.groupby('OUTCOME_CATEGORY')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
outcome_analysis.columns = ['Previous Outcome Pattern', 'Default Rate (%)', 'Count']
outcome_analysis = outcome_analysis.sort_values('Default Rate (%)', ascending=False)

no_prev_rate = status_analysis[status_analysis['has_prev_application'] == False]['default_rate'].values[0]
has_prev_rate = status_analysis[status_analysis['has_prev_application'] == True]['default_rate'].values[0]
never_approved_rate = approved_analysis[approved_analysis['has_approved'] == False]['default_rate'].values[0]
prev_approved_rate = approved_analysis[approved_analysis['has_approved'] == True]['default_rate'].values[0]

preprocessed_data['q4'] = {
    'status_analysis': status_analysis,
    'approved_analysis': approved_analysis,
    'outcome_analysis': outcome_analysis,
    'no_prev_rate': no_prev_rate,
    'has_prev_rate': has_prev_rate,
    'never_approved_rate': never_approved_rate,
    'prev_approved_rate': prev_approved_rate
}

# ============================================================
# D10: Credit Bureau Enquiry Frequency
# ============================================================
print("Processing D10: Credit Bureau Enquiry Frequency...")
enquiry_columns = [
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]

enquiry_data = app_data[['TARGET'] + enquiry_columns].copy()

enquiry_data['TOTAL_ENQUIRIES'] = enquiry_data[enquiry_columns].sum(axis=1)
enquiry_data['HAS_ENQUIRIES'] = (enquiry_data['TOTAL_ENQUIRIES'] > 0).astype(int)

total_enquiry_analysis = enquiry_data.groupby(enquiry_data['HAS_ENQUIRIES'].map({0: '0 Enquiries', 1: 'Non-zero Enquiries'}))['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
total_enquiry_analysis.columns = ['Total Enquiries', 'Default Rate (%)', 'Count']

if enquiry_data['HAS_ENQUIRIES'].std() > 0:
    corr_total = np.corrcoef(enquiry_data['HAS_ENQUIRIES'], enquiry_data['TARGET'])[0, 1]
else:
    corr_total = np.nan

zero_enq_rate = total_enquiry_analysis[total_enquiry_analysis['Total Enquiries'] == '0 Enquiries']['Default Rate (%)'].values[0] if '0 Enquiries' in total_enquiry_analysis['Total Enquiries'].values else None
non_zero_enq_rate = total_enquiry_analysis[total_enquiry_analysis['Total Enquiries'] == 'Non-zero Enquiries']['Default Rate (%)'].values[0] if 'Non-zero Enquiries' in total_enquiry_analysis['Total Enquiries'].values else None

preprocessed_data['q5'] = {
    'total_enquiry_analysis': total_enquiry_analysis,
    'corr_total': corr_total,
    'zero_enq_rate': zero_enq_rate,
    'non_zero_enq_rate': non_zero_enq_rate
}

# ============================================================
# E11: Loan Types (Cash vs Revolving)
# ============================================================
print("Processing E11: Loan Types (Cash vs Revolving)...")
contract_data = app_data[['TARGET', 'NAME_CONTRACT_TYPE']].copy()

contract_type_analysis = contract_data.groupby('NAME_CONTRACT_TYPE')['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count')
]).reset_index()
contract_type_analysis.columns = ['Contract Type', 'Default Rate (%)', 'Count']
contract_type_analysis = contract_type_analysis.sort_values('Default Rate (%)', ascending=False)

contract_distribution = contract_data['NAME_CONTRACT_TYPE'].value_counts().reset_index()
contract_distribution.columns = ['Contract Type', 'Count']

highest_risk_type = contract_type_analysis.iloc[0]
risk_diff_contract = contract_type_analysis.iloc[0]['Default Rate (%)'] - contract_type_analysis.iloc[-1]['Default Rate (%)']

preprocessed_data['loan_type_analysis'] = {
    'contract_type_analysis': contract_type_analysis,
    'contract_distribution': contract_distribution,
    'highest_risk_type': highest_risk_type,
    'risk_diff_contract': risk_diff_contract,
    'riskier_type': highest_risk_type['Contract Type']
}

# ============================================================
# E12: Loan Purpose Analysis
# ============================================================
print("Processing E12: Loan Purpose Analysis...")
# Using previous application data for loan purpose
if 'NAME_CASH_LOAN_PURPOSE' in prev_data.columns:
    purpose_data = prev_data[['SK_ID_CURR', 'NAME_CASH_LOAN_PURPOSE']].copy()
    purpose_data = purpose_data[purpose_data['NAME_CASH_LOAN_PURPOSE'].notna()].copy()
    
    # Merge with current application to get default status
    merged_purpose = app_data[['SK_ID_CURR', 'TARGET']].merge(purpose_data, on='SK_ID_CURR', how='inner')
    
    purpose_analysis = merged_purpose.groupby('NAME_CASH_LOAN_PURPOSE')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count')
    ]).reset_index()
    purpose_analysis.columns = ['Loan Purpose', 'Default Rate (%)', 'Count']
    purpose_analysis = purpose_analysis.sort_values('Default Rate (%)', ascending=False)
    purpose_analysis = purpose_analysis[purpose_analysis['Count'] >= 50]  # Minimum sample size
    
    top_risky_purposes = purpose_analysis.head(10)
    safest_purposes = purpose_analysis.tail(10)
    
    risk_range_purpose = purpose_analysis.iloc[0]['Default Rate (%)'] - purpose_analysis.iloc[-1]['Default Rate (%)'] if len(purpose_analysis) > 1 else 0
    
    preprocessed_data['loan_purpose_analysis'] = {
        'purpose_analysis': purpose_analysis,
        'top_risky_purposes': top_risky_purposes,
        'safest_purposes': safest_purposes,
        'risk_range_purpose': risk_range_purpose
    }
else:
    # If no purpose column, use goods category from previous application
    if 'NAME_GOODS_CATEGORY' in prev_data.columns:
        purpose_data = prev_data[['SK_ID_CURR', 'NAME_GOODS_CATEGORY']].copy()
        purpose_data = purpose_data[purpose_data['NAME_GOODS_CATEGORY'].notna()].copy()
        
        merged_purpose = app_data[['SK_ID_CURR', 'TARGET']].merge(purpose_data, on='SK_ID_CURR', how='inner')
        
        purpose_analysis = merged_purpose.groupby('NAME_GOODS_CATEGORY')['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        purpose_analysis.columns = ['Goods Category', 'Default Rate (%)', 'Count']
        purpose_analysis = purpose_analysis.sort_values('Default Rate (%)', ascending=False)
        purpose_analysis = purpose_analysis[purpose_analysis['Count'] >= 50]
        
        top_risky_purposes = purpose_analysis.head(10)
        safest_purposes = purpose_analysis.tail(10)
        
        risk_range_purpose = purpose_analysis.iloc[0]['Default Rate (%)'] - purpose_analysis.iloc[-1]['Default Rate (%)'] if len(purpose_analysis) > 1 else 0
        
        preprocessed_data['loan_purpose_analysis'] = {
            'purpose_analysis': purpose_analysis,
            'top_risky_purposes': top_risky_purposes,
            'safest_purposes': safest_purposes,
            'risk_range_purpose': risk_range_purpose
        }

# ============================================================
# F13: Social Circle Indicators
# ============================================================
print("Processing F13: Social Circle Indicators...")
social_columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 
                  'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']

if all(col in app_data.columns for col in social_columns):
    social_data = app_data[['TARGET'] + social_columns].copy()
    
    # Calculate default rates in social circle
    social_data['SOCIAL_30_DEFAULT_RATE'] = social_data['DEF_30_CNT_SOCIAL_CIRCLE'] / social_data['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    social_data['SOCIAL_60_DEFAULT_RATE'] = social_data['DEF_60_CNT_SOCIAL_CIRCLE'] / social_data['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    
    # Group by 30-day social circle default rate
    social_data['SOCIAL_30_GROUP'] = pd.cut(
        social_data['SOCIAL_30_DEFAULT_RATE'],
        bins=[-0.1, 0, 0.1, 0.3, 0.5, 1],
        labels=['0%', '0-10%', '10-30%', '30-50%', '50-100%']
    )
    
    social_30_analysis = social_data.groupby('SOCIAL_30_GROUP')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count')
    ]).reset_index()
    social_30_analysis.columns = ['Social Circle 30-Day Default Rate', 'Default Rate (%)', 'Count']
    social_30_analysis = social_30_analysis[social_30_analysis['Count'] >= 50]
    
    # Group by 60-day social circle default rate
    social_data['SOCIAL_60_GROUP'] = pd.cut(
        social_data['SOCIAL_60_DEFAULT_RATE'],
        bins=[-0.1, 0, 0.1, 0.3, 0.5, 1],
        labels=['0%', '0-10%', '10-30%', '30-50%', '50-100%']
    )
    
    social_60_analysis = social_data.groupby('SOCIAL_60_GROUP')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count')
    ]).reset_index()
    social_60_analysis.columns = ['Social Circle 60-Day Default Rate', 'Default Rate (%)', 'Count']
    social_60_analysis = social_60_analysis[social_60_analysis['Count'] >= 50]
    
    # Calculate correlations
    corr_30 = np.corrcoef(social_data['SOCIAL_30_DEFAULT_RATE'].fillna(0), social_data['TARGET'])[0, 1]
    corr_60 = np.corrcoef(social_data['SOCIAL_60_DEFAULT_RATE'].fillna(0), social_data['TARGET'])[0, 1]
    
    preprocessed_data['social_circle_analysis'] = {
        'social_30_analysis': social_30_analysis,
        'social_60_analysis': social_60_analysis,
        'corr_30_days': corr_30,
        'corr_60_days': corr_60
    }

# ============================================================
# F14: Document Submission Behavior
# ============================================================
print("Processing F14: Document Submission Behavior...")
# Get all document flag columns
doc_columns = [col for col in app_data.columns if col.startswith('FLAG_DOCUMENT_')]

if doc_columns:
    doc_data = app_data[['TARGET'] + doc_columns].copy()
    
    # Calculate total documents submitted
    doc_data['TOTAL_DOCS'] = doc_data[doc_columns].sum(axis=1)
    
    # Group by number of documents
    doc_data['DOCS_GROUP'] = pd.cut(
        doc_data['TOTAL_DOCS'],
        bins=[-1, 0, 2, 5, 8, 15, float('inf')],
        labels=['0', '1-2', '3-5', '6-8', '9-15', '16+']
    )
    
    documents_submitted_analysis = doc_data.groupby('DOCS_GROUP')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count'),
        ('mean_docs', lambda x: doc_data.loc[x.index, 'TOTAL_DOCS'].mean())
    ]).reset_index()
    documents_submitted_analysis.columns = ['Documents Submitted', 'Default Rate (%)', 'Count', 'Mean Documents']
    
    corr_documents = np.corrcoef(doc_data['TOTAL_DOCS'], doc_data['TARGET'])[0, 1]
    
    # Analyze specific documents that have biggest impact
    specific_docs_analysis = []
    for doc_col in doc_columns:
        doc_submission = doc_data.groupby(doc_col)['TARGET'].agg([
            ('default_rate', lambda x: (x.sum() / len(x)) * 100),
            ('count', 'count')
        ]).reset_index()
        doc_submission.columns = ['Submitted', 'Default Rate (%)', 'Count']
        
        if len(doc_submission) == 2:  # Binary submission (0 or 1)
            submitted_rate = doc_submission[doc_submission['Submitted'] == 1]['Default Rate (%)'].values[0]
            not_submitted_rate = doc_submission[doc_submission['Submitted'] == 0]['Default Rate (%)'].values[0]
            diff = submitted_rate - not_submitted_rate
            
            specific_docs_analysis.append({
                'Document Type': doc_col.replace('FLAG_DOCUMENT_', 'Doc '),
                'Default Rate Difference (%)': diff,
                'Submitted Rate': submitted_rate,
                'Not Submitted Rate': not_submitted_rate
            })
    
    specific_docs_df = pd.DataFrame(specific_docs_analysis)
    specific_docs_df = specific_docs_df.sort_values('Default Rate Difference (%)')
    
    no_docs_rate = documents_submitted_analysis.iloc[0]['Default Rate (%)']
    max_docs_row = documents_submitted_analysis.loc[documents_submitted_analysis['Default Rate (%)'].idxmin()]
    max_docs_rate = max_docs_row['Default Rate (%)']
    
    preprocessed_data['document_analysis'] = {
        'documents_submitted_analysis': documents_submitted_analysis,
        'specific_docs_analysis': specific_docs_df,
        'corr_documents': corr_documents,
        'no_docs_rate': no_docs_rate,
        'max_docs_rate': max_docs_rate
    }

# ============================================================
# F15: Region Ratings Analysis
# ============================================================
print("Processing F15: Region Ratings Analysis...")
if 'REGION_RATING_CLIENT' in app_data.columns:
    region_data = app_data[['TARGET', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].copy()
    
    # Client region rating analysis
    region_client_analysis = region_data.groupby('REGION_RATING_CLIENT')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count')
    ]).reset_index()
    region_client_analysis.columns = ['Region Rating', 'Default Rate (%)', 'Count']
    region_client_analysis = region_client_analysis.sort_values('Region Rating')
    
    # City region rating analysis
    region_city_analysis = region_data.groupby('REGION_RATING_CLIENT_W_CITY')['TARGET'].agg([
        ('default_rate', lambda x: (x.sum() / len(x)) * 100),
        ('count', 'count')
    ]).reset_index()
    region_city_analysis.columns = ['Region Rating (with City)', 'Default Rate (%)', 'Count']
    region_city_analysis = region_city_analysis.sort_values('Region Rating (with City)')
    
    corr_client_rating = np.corrcoef(region_data['REGION_RATING_CLIENT'], region_data['TARGET'])[0, 1]
    corr_city_rating = np.corrcoef(region_data['REGION_RATING_CLIENT_W_CITY'], region_data['TARGET'])[0, 1]
    
    worst_rating_client = region_client_analysis.loc[region_client_analysis['Default Rate (%)'].idxmax()]
    risk_range_region = region_client_analysis['Default Rate (%)'].max() - region_client_analysis['Default Rate (%)'].min()
    
    preprocessed_data['region_analysis'] = {
        'region_client_analysis': region_client_analysis,
        'region_city_analysis': region_city_analysis,
        'corr_client_rating': corr_client_rating,
        'corr_city_rating': corr_city_rating,
        'worst_rating_client': worst_rating_client,
        'risk_range_region': risk_range_region
    }

print("Saving preprocessed data...")
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("Preprocessing complete! Data saved to preprocessed_data.pkl")
print(f"Total analyses processed: {len(preprocessed_data)}")