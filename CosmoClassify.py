"""
# AMAL ALKRAIMEEN
# Cosmo Classify Project

# Main Purpose:
# This Python script performs a cosmo classification analysis using machine learning techniques. The main goals of this project are as follows:
# 1. Load and explore a dataset containing information about cosmological objects.
# 2. Preprocess the data by encoding categorical variables and addressing class imbalance.
# 3. Train and evaluate various machine learning models to classify cosmological objects based on their properties.
# 4. Compare the performance of different models in terms of recall score.
# 5. Provide a structured and modular codebase for future enhancements and model comparisons.

# Project Structure:
# - The code is organized into functions, each responsible for a specific part of the analysis.
# - The main program section at the end of the code orchestrates the entire analysis process.
# - Models such as Logistic Regression, K Nearest Neighbors, Decision Tree, Gaussian Naive Bayes, and Random Forest
#   are trained and evaluated for cosmo classification.
If you want to run this code in your device make sure you adjust the 'file_path' variable to the actual path of your cosmo classification dataset.
Let's Enjoy space!!!
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def read_data(file_path):
   
    df = pd.read_csv('C:/Users/user/Desktop/Cosmo classify/star_classification.csv')
    return df

def perform_eda(df):
    # EDA (Exploratory Data Analysis)
    df_info = df.info()
    df_desc = df.describe()
    unique_classes = df['class'].nunique()
    class_counts = df['class'].value_counts()

    sns.set(style='darkgrid', palette='dark')
    sns.countplot(x=df['class'])
    plt.show()

    sns.distplot(df['delta'], color="midnightblue")
    plt.show()

    sns.distplot(df['redshift'], color="midnightblue")
    plt.show()

    sns.distplot(df['plate'], color="midnightblue")
    plt.show()

    sns.scatterplot(x=df['alpha'], y=df['delta'], color="midnightblue")
    plt.show()

    df.hist(bins=25, figsize=(14, 14), color="midnightblue")
    plt.show()

    plt.figure(figsize=(15, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True)

def encode_data(df):
    # Encoding
    LE = LabelEncoder()
    df['class'] = LE.fit_transform(df['class'])
    X = df[['u', 'g', 'r', 'i', 'z', 'redshift', 'plate']]
    y = df['class']
    return X, y

def perform_resampling(X, y):
    # Resampling
    sm = SMOTE(random_state=30, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def split_data(X_res, y_res):
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=30)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    # Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    # K Neighbors Classifier
    knn_df = pd.DataFrame(columns=['Neighbors', 'Recall score'])
    for i in range(1, 21):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        y_pred2 = model.predict(X_test)
        knn_df = knn_df.append({'Neighbors': i, 'Recall score': recall_score(y_test, y_pred2, average='weighted')}, ignore_index=True)
    knn_df = knn_df.sort_values(by='Recall score', ascending=False)
    best_k = knn_df.iloc[0]['Neighbors']
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    # Decision Tree
    model = DecisionTreeClassifier(random_state=30)
    model.fit(X_train, y_train)
    return model

def train_gaussian_nb(X_train, y_train):
    # Gaussian Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    # Random Forest
    rf_df = pd.DataFrame(columns=['Estimators', 'Recall score'])
    for i in range(1, 21):
        model = RandomForestClassifier(n_estimators=i, random_state=30)
        model.fit(X_train, y_train)
        y_pred5 = model.predict(X_test)
        rf_df = rf_df.append({'Estimators': i, 'Recall score': recall_score(y_test, y_pred5, average='weighted')}, ignore_index=True)
    rf_df = rf_df.sort_values(by='Recall score', ascending=False)
    best_estimators = rf_df.iloc[0]['Estimators']
    model = RandomForestClassifier(n_estimators=best_estimators, random_state=30)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred, average='weighted')
    return recall

if __name__ == "__main__":
   
    file_path = 'C:/Users/user/Desktop/Cosmo classify/star_classification.csv'
    
    df = read_data(file_path)
    perform_eda(df)
    
    X, y = encode_data(df)
    X_res, y_res = perform_resampling(X, y)
    X_train, X_test, y_train, y_test = split_data(X_res, y_res)
    
    logistic_regression_model = train_logistic_regression(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    decision_tree_model = train_decision_tree(X_train, y_train)
    gaussian_nb_model = train_gaussian_nb(X_train, y_train)
    random_forest_model = train_random_forest(X_train, y_train)
    
    models = {
        'Logistic Regression': logistic_regression_model,
        'K Nearest Neighbors': knn_model,
        'Decision Tree': decision_tree_model,
        'Gaussian Naive Bayes': gaussian_nb_model,
        'Random Forest': random_forest_model
    }
    
    for model_name, model in models.items():
        recall = evaluate_model(model, X_test, y_test)
        print(f'{model_name} Recall Score: {recall}')
