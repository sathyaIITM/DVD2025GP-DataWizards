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

print("Processing Q1: Income & Credit Ratios...")
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

print("Processing Q2: Employment Tenure Impact...")
tenure_data = app_data[['TARGET', 'DAYS_EMPLOYED']].copy()
tenure_data = tenure_data[tenure_data['DAYS_EMPLOYED'].notna()].copy()

tenure_data['YEARS_EMPLOYED'] = -tenure_data['DAYS_EMPLOYED'] / 365.25
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

tenure_bins = pd.cut(tenure_data['YEARS_EMPLOYED'], bins=20)
tenure_bin_analysis = tenure_data.groupby(tenure_bins)['TARGET'].agg([
    ('default_rate', lambda x: (x.sum() / len(x)) * 100),
    ('count', 'count'),
    ('mean_years', lambda x: tenure_data.loc[x.index, 'YEARS_EMPLOYED'].mean())
]).reset_index()
tenure_bin_analysis.columns = ['Years Range', 'Default Rate (%)', 'Count', 'Mean Years']
tenure_bin_analysis = tenure_bin_analysis[tenure_bin_analysis['Count'] >= 100]

shortest_tenure = tenure_category_analysis.iloc[0]['Default Rate (%)']
longest_tenure = tenure_category_analysis.iloc[-1]['Default Rate (%)']
diff_tenure = shortest_tenure - longest_tenure

preprocessed_data['q2'] = {
    'tenure_category_analysis': tenure_category_analysis,
    'tenure_quintile_analysis': tenure_quintile_analysis,
    'tenure_bin_analysis': tenure_bin_analysis,
    'corr_tenure': corr_tenure,
    'shortest_tenure': shortest_tenure,
    'longest_tenure': longest_tenure,
    'diff_tenure': diff_tenure
}

print("Processing Q3: Employment Types & Occupations...")
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

print("Processing Q4: Previous Loan Outcomes...")
prev_status_data = prev_data[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].copy()

prev_status_summary = prev_status_data.groupby('SK_ID_CURR')['NAME_CONTRACT_STATUS'].agg([
    ('has_approved', lambda x: (x == 'Approved').any()),
    ('has_refused', lambda x: (x == 'Refused').any()),
    ('has_canceled', lambda x: (x == 'Canceled').any()),
    ('has_unused', lambda x: (x == 'Unused offer').any()),
    ('total_prev', 'count'),
    ('approved_count', lambda x: (x == 'Approved').sum()),
    ('refused_count', lambda x: (x == 'Refused').sum()),
    ('canceled_count', lambda x: (x == 'Canceled').sum())
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

print("Processing Q5: Credit Bureau Enquiry Frequency...")
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

print("Saving preprocessed data...")
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump(preprocessed_data, f)

print("Preprocessing complete! Data saved to preprocessed_data.pkl")

