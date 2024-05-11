# Important Librarys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from imblearn.over_sampling import SMOTE

# Data loading
df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

#Data Preprocessing

# Converting Yes to 1 and No to 0
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

# Droping use less columns
df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis="columns", inplace=True)


#One Hot Encoding
ohe = OneHotEncoder()
#Separating all categorical data from the dataset
X_cat = df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat = ohe.fit_transform(X_cat).toarray()
X_cat = pd.DataFrame(X_cat)
# Here i will drop Some feature who highly corelated with each other
#'MonthlyIncome','YearsWithCurrManager','YearsInCurrentRole','TotalWorkingYears',
#assigning column names
X_cat.columns = ohe.get_feature_names_out(['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'])
X_num = df[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
              'JobLevel', 'JobSatisfaction', 'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike', 
              'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'TrainingTimesLastYear', 'WorkLifeBalance', 
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


# #Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
grid_search_params = {
    'penalty': ['l1', 'l2'],  # Regularization penalty (l1: Lasso, l2: Ridge)
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algorithm to use in optimization problem
    'max_iter': [100, 150, 200, 250, 300],
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


# Saving Model
pickle.dump(model, open("final_model.pkl", "wb"))