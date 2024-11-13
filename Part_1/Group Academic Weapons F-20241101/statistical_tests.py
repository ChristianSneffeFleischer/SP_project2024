import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import chi2_contingency, kruskal
from statsmodels.discrete.discrete_model import MNLogit
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel files
private_data_path = 'private_dataF.xlsx'  
public_data_register_path = 'public_data_registerF.xlsx'  
public_data_results_path = 'public_data_resultsF.xlsx'

private_data = pd.read_excel(private_data_path, engine='openpyxl')
public_data_register = pd.read_excel(public_data_register_path, engine='openpyxl')
public_data_results = pd.read_excel(public_data_results_path, engine='openpyxl')

# Calculate age from dob
df['age'] = df['dob'].apply(lambda x: datetime.now().year - x.year)

# Function to convert categorical data to numeric
def encode_categories(column):
    return pd.Categorical(df[column]).codes

# Encode categorical variables for statistical analysis
df['sex_encoded'] = encode_categories('sex')
df['party_encoded'] = encode_categories('party')
df['marital_status_encoded'] = encode_categories('marital_status')
df['education_encoded'] = encode_categories('education')
df['citizenship_encoded'] = encode_categories('citizenship')

# ---- Chi-Square Tests for Categorical Variables ----
# Gender vs. Party Preference
contingency_table_sex_party = pd.crosstab(df['sex'], df['party'])
chi2_sex, p_value_sex_party, _, _ = chi2_contingency(contingency_table_sex_party)

# Marital Status vs. Party Preference
contingency_table_marital_party = pd.crosstab(df['marital_status'], df['party'])
chi2_marital, p_value_marital_party, _, _ = chi2_contingency(contingency_table_marital_party)

# Citizenship vs. Party Preference
contingency_table_citizenship_party = pd.crosstab(df['citizenship'], df['party'])
chi2_citizenship, p_value_citizenship_party, _, _ = chi2_contingency(contingency_table_citizenship_party)

# Education vs. Party Preference
contingency_table_education_party = pd.crosstab(df['education'], df['party'])
chi2_education, p_value_education_party, _, _ = chi2_contingency(contingency_table_education_party)

# ---- Display Results ----
print("Chi-Square Test Results:")
print(f"Gender vs. Party Preference: Chi2 = {chi2_sex}, p-value = {p_value_sex_party}")
print(f"Marital Status vs. Party Preference: Chi2 = {chi2_marital}, p-value = {p_value_marital_party}")
print(f"Citizenship vs. Party Preference: Chi2 = {chi2_citizenship}, p-value = {p_value_citizenship_party}")
print(f"Education vs. Party Preference: Chi2 = {chi2_education}, p-value = {p_value_education_party}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel file
file_path = 'private_dataF.xlsx'  # Update this path if necessary
df = pd.read_excel(file_path, engine='openpyxl')

custom_palette = sns.color_palette("Set2")

# Create a figure and a set of subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 2x2 grid of subplots

# Plot: Distribution of Votes by Sex
sns.countplot(x='party', hue='sex', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Votes by Gender')
axes[0, 0].set_xlabel('Party')
axes[0, 0].set_ylabel('Count')

# Plot: Distribution of Votes by Marital Status
sns.countplot(x='party', hue='marital_status', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Votes by Marital Status')
axes[0, 1].set_xlabel('Party')
axes[0, 1].set_ylabel('Count')

# Plot: Distribution of Votes by Education Level
sns.countplot(x='party', hue='education', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Votes by Education Level')
axes[1, 0].set_xlabel('Party')
axes[1, 0].set_ylabel('Count')

# Plot: Distribution of Votes by Citizenship
df_filtered = df[df['citizenship'] != 'Denmark']
sns.countplot(x='party', hue='citizenship', data=df_filtered, ax=axes[1, 1])
axes[1, 1].set_title('Distribution of Votes by Minorities\' Citizenship')
axes[1, 1].set_xlabel('Party')
axes[1, 1].set_ylabel('Count')

# Adjust the layout to prevent overlapping
plt.tight_layout()
plt.show()
