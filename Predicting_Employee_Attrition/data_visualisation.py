# Important Librarys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu


# Data loading
df=pd.read_csv('datasets/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Converting Yes to 1 and No to 0
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

sns.countplot(x='Attrition', data=df)
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.xticks([0,1], ['Stayed', 'Left'])
plt.annotate(f'{df["Attrition"].value_counts()[0]} \n {round(df["Attrition"].value_counts()[0]/len(df)*100,1)} %', (0, df["Attrition"].value_counts()[0]), ha='center', va='bottom')
plt.annotate(f'{df["Attrition"].value_counts()[1]}\n {round(df["Attrition"].value_counts()[1]/len(df)*100,1)}%', (1, df["Attrition"].value_counts()[1]), ha='center', va='bottom')
plt.show()


df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis="columns", inplace=True)
left_df = df[df['Attrition'] == 1]
stayed_df = df[df['Attrition'] == 0]
print(f'Total Employees: {len(df)}\n')
print(f'Number of employees who left: {df["Attrition"].value_counts()[1]}')
print(f'% of employees who left: {round(df["Attrition"].value_counts()[1]/len(df)*100,2)}%\n')
print(f'Number of employees who stayed: {df["Attrition"].value_counts()[0]}')
print(f'% of employees who stayed: {round(df["Attrition"].value_counts()[0]/len(df)*100,2)}%')

left_df = df[df['Attrition'] == 1]
stayed_df = df[df['Attrition'] == 0]

print(stayed_df.describe())
print(left_df.describe())
#  Let's compare the mean and std of the employees who stayed and left
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level


# Difining new dataset for Categrical EDA
df1=df[['BusinessTravel',
 'Department',
 'EducationField',
 'JobRole',
 'MaritalStatus',
 'OverTime',
 'Education',
 'JobInvolvement',
 'JobLevel']]

#Exploratory Data Analysis
#Univariate Analysis
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        plt.figure(figsize=(8,5))
        sns.histplot(df[column],kde=True,color='Purple')
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
        plt.tight_layout()
    # For categorical columns, plot a countplot
    else:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, data=df, palette='viridis')
        plt.title(f'Countplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()


#plot histogram of numeric df with Attrition 1 or 0
fig, ax = plt.subplots(1, 3, figsize=(20, 6))
sns.histplot(df, x='Age', hue='Attrition', kde=True, ax=ax[0])
sns.histplot(df, x='DailyRate', hue='Attrition', kde=True, ax=ax[1])
sns.histplot(df, x='DistanceFromHome', hue='Attrition', kde=True, ax=ax[2])

fig, ax = plt.subplots(1, 3, figsize=(20, 6))
sns.histplot(df, x='EnvironmentSatisfaction', hue='Attrition', kde=True, ax=ax[0])
sns.histplot(df, x='JobSatisfaction', hue='Attrition', kde=True, ax=ax[1])
sns.histplot(df, x='StockOptionLevel', hue='Attrition', kde=True, ax=ax[2])

correlations = df.corr(numeric_only=True, method='spearman') #We're using Spearman's Correlation Coefficient as we are dealing with non-parametric df (not normally distributed)
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(correlations, annot=True)
# Job level is strongly correlated with total working hours
# Monthly income is strongly correlated with Job level
# Monthly income is strongly correlated with total working hours
# Age is stongly correlated with monthly income
# Plotting how every  categorical feature correlate with the "target"
for column in df1:
    plt.figure(figsize=(8,5), facecolor='white')
    sns.histplot(x=df1[column]           #plotting count plot
                    ,hue=df.Attrition,
                    palette={1: "red", 0: "blue"})
    plt.xlabel(column,fontsize=20)#assigning name to x-axis and increasing it's font
    plt.ylabel('Attrition',fontsize=20)
    plt.tight_layout()
    plt.show()

# Single employees tend to leave compared to married and divorced
# Sales Representitives tend to leave compared to any other job
# Less involved employees tend to leave the company
# Less experienced (low job level) tend to leave the company

# There is significant difference in the distance from home between employees who left and stayed (p<.05)
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['DistanceFromHome'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['DistanceFromHome'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Distance From Home')
plt.legend()
plt.show()

#Mann-Whitney's test to check if there is a significant difference between the two groups
stats, p = mannwhitneyu(left_df["DistanceFromHome"], stayed_df["DistanceFromHome"])
print(f'p-value: {p}')
# p-value is 0.0023870470273627984 which is less than 0.05, so we reject the null hypothesis
# There is significant difference in the distance from home between employees who left and stayed
# There is significant difference in the Years With Current Manager between employees who left and stayed (p<.05)
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['YearsWithCurrManager'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Years With Current Manager')
plt.legend()
plt.show()

#Mann-Whitney's test to check if there is a significant difference between the two groups
stats, p = mannwhitneyu(left_df["YearsWithCurrManager"], stayed_df["YearsWithCurrManager"])
print(f'p-value: {p}')
# p-value is 1.8067542583144407e-11 which is less than 0.05, so we reject the null hypothesis
# There is significant difference in the years with current manager between employees who left and stayed
# There is significant difference in the Total Working Years between employees who left and stayed (p<.05)
plt.figure(figsize=(12,7))
sns.kdeplot(left_df['TotalWorkingYears'], label='Employees who left', fill=True, color='r')
sns.kdeplot(stayed_df['TotalWorkingYears'], label='Employees who Stayed', fill=True, color='b')
plt.xlabel('Total Working Years')
plt.legend()
plt.show()

#Mann-Whitney's test to check if there is a significant difference between the two groups
stats, p = mannwhitneyu(left_df["TotalWorkingYears"], stayed_df["TotalWorkingYears"])
print(f'p-value: {p}')
# p-value is 2.399569364798952e-14 which is less than 0.05, so we reject the null hypothesis
# There is significant difference in the total working years between employees who left and stayed
# There are no significant differences in Monthly Income between Female and Male employees (p=0.09)
# Let's see the Gender vs. Monthly Income
sns.boxplot(x='Gender', y='MonthlyIncome', data=df)

#Mann-Whitney's test to check if there is a significant difference between Male and Female MonthlyIncome
male_income = df[df['Gender'] == 'Male']['MonthlyIncome']
female_income = df[df['Gender'] == 'Female']['MonthlyIncome']

stats, p = mannwhitneyu(male_income, female_income)
print(f'p-value: {p}')
# p-value is 0.08841668326602112 which is greater than 0.05, so we fail to reject the null hypothesis and assume no differences in MonthlyIncome between Male and Female employees
# Research Directors and Managers have the highest Monthly Income
# Sales Representatives have the lowest Monthly Income, followed by Research Scientists and Lab Technicians

# Let's see the Job Role vs. Monthly Income
plt.figure(figsize=(15, 10))
sns.boxplot(x='MonthlyIncome', y='JobRole', data=df)
plt.show()