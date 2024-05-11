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








model = pickle.load(open("model.pkl", "rb"))