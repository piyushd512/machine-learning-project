# Important Librarys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE

# Data loading
df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

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

#Mann-Whitney's test to check if there is a significant difference between the two groups
from scipy.stats import mannwhitneyu
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

#Mann-Whitney's test to check if there is a significant difference between the two groups
from scipy.stats import mannwhitneyu
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

#Mann-Whitney's test to check if there is a significant difference between the two groups
from scipy.stats import mannwhitneyu
stats, p = mannwhitneyu(left_df["TotalWorkingYears"], stayed_df["TotalWorkingYears"])
print(f'p-value: {p}')
# p-value is 2.399569364798952e-14 which is less than 0.05, so we reject the null hypothesis
# There is significant difference in the total working years between employees who left and stayed
# There are no significant differences in Monthly Income between Female and Male employees (p=0.09)
# Let's see the Gender vs. Monthly Income
sns.boxplot(x='Gender', y='MonthlyIncome', data=df)

#Mann-Whitney's test to check if there is a significant difference between Male and Female MonthlyIncome
from scipy.stats import mannwhitneyu
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
ohe = OneHotEncoder()
#Separating all categorical data from the dataset
X_cat = df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat = ohe.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
# Here i will drop Some feature who highly corelated with each other
#'MonthlyIncome','YearsWithCurrManager','YearsInCurrentRole'
#assigning column names
X_cat.columns = ohe.get_feature_names_out(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'])
X_num = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
              'JobLevel', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
              'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
              'YearsAtCompany', 'YearsSinceLastPromotion']]

X_all = pd.concat([X_num, X_cat], axis=1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_all)

## Creating Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['Attrition'], test_size=0.25,random_state=42)

## Using SMOTE algorithm for class balancing
smote = SMOTE(random_state=42, sampling_strategy='minority')
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

## Converting train test to PyTorch Tensor
X_train_t=torch.tensor(X_train_smote, dtype=torch.float32)
X_test_t=torch.tensor(X_test, dtype=torch.float32)
y_train_t=torch.tensor(y_train_smote.values, dtype=torch.float32).view(-1, 1)
y_test_t=torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


## Testing diffrent Models
models=[LogisticRegression(),SVC(),RandomForestClassifier(),DecisionTreeClassifier(),XGBClassifier(),CatBoostClassifier(verbose=0),KNeighborsClassifier(),GaussianNB(),GradientBoostingClassifier(),AdaBoostClassifier()]
train_acc=[]
test_acc=[]
recall_scores=[]
precision_scores=[]
f1_scores=[]

# Loop through the models
for model in models:
    model.fit(X_train_smote, y_train_smote)
    # prediction with x_test
    y_pred=model.predict(X_test)
    #prediction with x_train
    y_train_predict=model.predict(X_train_smote)

    train_accuracy=accuracy_score(y_train_smote,y_train_predict)
    train_acc.append(train_accuracy)
    test_accuracy=accuracy_score(y_test,y_pred)
    test_acc.append(test_accuracy)
    recall=recall_score(y_test,y_pred)
    recall_scores.append(recall)
    precision=precision_score(y_test,y_pred)
    precision_scores.append(precision)
    f1=f1_score(y_test,y_pred)
    f1_scores.append(f1)
    
    print(f"Accuracy score of {model} is: {train_accuracy} and {test_accuracy}")
    print(f"F1 score of {model} is: {f1}")
    print(f"Precision score of {model} is: {precision}")
    print(f"Recall score of {model} is: {recall}")
    cb_cnf = confusion_matrix(y_test, y_pred)
    sns.heatmap(cb_cnf, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model}  Confusion Matrix')
    plt.show()
    print(f"Classification report of {model} is: {classification_report(y_test,y_pred)}")
    print("*"*76)

class SimpleNNBinary(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNNBinary, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer to hidden layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
model = SimpleNNBinary(input_size=X_train_t.shape[1], hidden_size=32, output_size=1)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_t)
    # Compute loss
    loss = criterion(outputs, y_train_t)
    # Zero gradients, backward pass, and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Print progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()  # Set model to evaluation mode
    test_outputs = model(X_test_t)
    train_outputs = model(X_train_t)
    # Convert probabilities to binary predictions
    predictions_train=(train_outputs>=0.5).float()
    predictions = (test_outputs >= 0.5).float()
    accuracy_train = accuracy_score(y_train_t, predictions_train)
    train_acc.append(accuracy_train)
    accuracy_test = accuracy_score(y_test_t, predictions)
    test_acc.append(accuracy_test)
    recall = recall_score(y_test_t, predictions)
    recall_scores.append(recall)
    precision=precision_score(y_test,predictions)
    precision_scores.append(precision)
    f1=f1_score(y_test,predictions)
    f1_scores.append(f1)

    print(f'Accuracy on training set: {accuracy_train*100:.2f}%')
    print(f'Accuracy on test set: {accuracy_test*100:.2f}%')
    print(f'Recall on test set: {recall*100:.2f}%')
    print(f"F1 score  is: {f1}")
    print(f"Precision score is: {precision}")
    cb_cnf = confusion_matrix(y_test, predictions)
    sns.heatmap(cb_cnf, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Classifier Confusion Matrix')
    plt.show()
    print(f"Classification report of is: {classification_report(y_test,predictions)}")


mods=['LogisticRegression','SVC','RandomForestClassifier','DecisionTreeClassifier','XGBClassifier','CatBoostClassifier','KNeighborsClassifier','GaussianNB','GradientBoostingClassifier','AdaBoostClassifier','SimpleNNBinary']


plt.figure(figsize=(10, 6))

# Train accuracy
plt.plot(mods, train_acc, marker='o', label='Train Accuracy')

# Test accuracy
plt.plot(mods, test_acc, marker='o', label='Test Accuracy')

# Recall score
plt.plot(mods, recall_scores, marker='o', label='Recall Score')

# f1 score
plt.plot(mods, f1_scores, marker='o', label='F1 Score')

# Precesion Score
plt.plot(mods, precision_scores,marker='o', label='Precision Score')
# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparison of Models')
plt.legend()

# Show plot
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


from sklearn.model_selection import GridSearchCV
grid_search_params = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (l1: Lasso, l2: Ridge)
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in optimization problem
    'max_iter': [100, 150, 200, 250, 300],  # Maximum number of iterations for optimization
    'class_weight': [None, 'balanced'],  # Weights associated with classes to address class imbalance
    'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria
    'fit_intercept': [True, False],  # Whether to fit the intercept for the model
    'multi_class': ['auto', 'ovr', 'multinomial'],  # Strategy for multiclass classification
    'warm_start': [True, False],  # Whether to reuse the solution of the previous call to fit as initialization
}



lo_clf = LogisticRegression(random_state=3)
lo_cv = GridSearchCV(lo_clf, grid_search_params, scoring="recall", n_jobs=-1, verbose=1, cv=5)
lo_cv.fit(X_train_smote, y_train_smote)
best_params = lo_cv.best_params_
print(f"Best paramters: {best_params})")

model=LogisticRegression(**best_params)
model.fit(X_train_smote, y_train_smote)#training the model
y_pred=model.predict(X_test)

#prediction with x_train
y_train_predict=model.predict(X_train_smote)
train_accuracy=accuracy_score(y_train_smote,y_train_predict)
test_accuracy=accuracy_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)

print(f"Accuracy score of {model} is: {train_accuracy} and {test_accuracy}")
print(f"F1 score of {model} is: {f1_score(y_test,y_pred)}")
print(f"Precision score of {model} is: {precision_score(y_test,y_pred)}")
print(f"Recall score of {model} is: {recall_score(y_test,y_pred)}")
cb_cnf = confusion_matrix(y_test, y_pred)
sns.heatmap(cb_cnf, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'{model} Classifier Confusion Matrix')
plt.show()
print(f"Classification report of {model} is: {classification_report(y_test,y_pred)}")
print("*"*76)