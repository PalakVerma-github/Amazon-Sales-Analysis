
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, confusion_matrix


df = pd.read_csv("C:/Users/Rangi/OneDrive/Desktop/Python Project/Amazon India Sales Data_Python.csv")


# Gathering Information related to dataset
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Ensure Amount is numeric
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')


# DATA CLEANING
# Fill NAs
df.fillna("Unknown", inplace=True)
#Convert date column
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
#Remove duplicates
df.drop_duplicates(inplace=True)
#Fix column names 
df.columns = df.columns.str.strip().str.lower()



# Ensure cleaning
print(df.isnull().sum())
print(df.dtypes)
print("Duplicates:", df.duplicated().sum())
print(df['amount'].describe())
print(df['status'].unique())
print(df['category'].unique())
print(df.shape)


#VISUALIZATIONS
# 1. Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['amount'], bins=30, color='orange', kde=True)
plt.title("Sales Amount Distribution", fontsize=14)
plt.show()

# 2. Category-wise Sales
plt.figure(figsize=(10,5))
sns.barplot(x='category', y='amount', data=df, palette='magma')
plt.xticks(rotation=45)
plt.title("Category-wise Sales", fontsize=14)
plt.show()

# 3. Monthly Sales Trend
if 'Date' in df.columns:
    df['Month'] = df['date'].dt.month

    monthly_sales = df.groupby('Month')['amount'].sum()

    plt.figure(figsize=(10,5))
    monthly_sales.plot(marker='o', color='red')
    plt.title("Monthly Sales Trend", fontsize=14)
    plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5.Top 10 States by Sales
if 'ship-state' in df.columns:
    top_states = df.groupby('ship-state')['amount'].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10,5))
    top_states.plot(kind='bar', color='pink')
    plt.title("Top 10 States by Sales")
    plt.show()

# 6. LINEAR REGRESSION MODEL (Predicting Sales Amount)
# Select features (adjust if needed)
features = ['qty']  # you can add more columns if available

X = df[features]
y = df['amount']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df['amount'] = df['amount'].fillna(df['amount'].median())
# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

#Prediction Visualization
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Sales")
plt.show()

#HYPOTHESIS TESTING
# 1. T-Test (B2B vs B2C Sales)
if 'B2B' in df.columns:
    group1 = df[df['B2B'] == True]['Amount']
    group2 = df[df['B2B'] == False]['Amount']

    t_stat, p_val = stats.ttest_ind(group1, group2, nan_policy='omit')

    print("T-test p-value:", p_val)
    
# 2. Chi-Square Test (Category vs Status)
if 'Category' in df.columns and 'Status' in df.columns:
    contingency = pd.crosstab(df['Category'], df['Status'])

    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print("Chi-square p-value:", p)
    
    
# 3. Normality Test (Shapiro Test)
stat, p = stats.shapiro(df['Amount'].sample(500))
print("Shapiro Test p-value:", p)
