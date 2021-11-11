import sys
!{sys.executable} -m pip install -U pandas-profiling[notebook]
!jupyter nbextension enable --py widgetsnbextension
!pip install matplotlib
!pip install graphviz

from google.colab import files
uploaded = files.upload()

import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=arff.loadarff("german_credit.arff")

df=pd.DataFrame(data[0])
for col in df.columns:
  if df[col].dtype=='object':
    df[col] = df[col].str.decode('utf-8')
print(df.dtypes)



df.head(10)

#Checking for any null/missing values
df.isnull().sum()

df.describe()

boxplot = df.boxplot(column="Credit Amount")


boxplot = df.boxplot(column="No of dependents")

boxplot = df.boxplot(column="Age (years)")

boxplot = df.boxplot(column="Duration of Credit (month)")

boxplot = df.boxplot(column="No of Credits at this Bank")

boxplot = df.boxplot(column="Instalment per cent")

df.hist(column="Instalment per cent")

df.hist(column="No of Credits at this Bank")

df.hist(column="Age (years)")

df.hist(column="No of dependents")

df.hist(column="Credit Amount")

df.hist(column="Duration of Credit (month)")

corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)

#Converting certain columns to numeric in order to generate histograms and correlation heatmap of variable
df["Creditability"] = pd.to_numeric(df["Creditability"])
df.hist(column="Creditability")

df["Account Balance"] = pd.to_numeric(df["Account Balance"])
df.hist(column="Account Balance")

df["Payment Status of Previous Credit"] = pd.to_numeric(df["Payment Status of Previous Credit"])
df.hist(column="Payment Status of Previous Credit")

df["Purpose"] = pd.to_numeric(df["Purpose"])
df.hist(column="Purpose")

df["Value Savings/Stocks"] = pd.to_numeric(df["Value Savings/Stocks"])
df.hist(column="Value Savings/Stocks")

df["Length of current employment"] = pd.to_numeric(df["Length of current employment"])
df.hist(column="Length of current employment")

df["Instalment per cent"] = pd.to_numeric(df["Instalment per cent"])
df.hist(column="Instalment per cent")

df["Sex & Marital Status"] = pd.to_numeric(df["Sex & Marital Status"])
df.hist(column="Sex & Marital Status")

df["Guarantors"] = pd.to_numeric(df["Guarantors"])
df.hist(column="Guarantors")

df["Duration in Current address"] = pd.to_numeric(df["Duration in Current address"])
df.hist(column="Duration in Current address")

df["Most valuable available asset"] = pd.to_numeric(df["Most valuable available asset"])
df.hist(column="Most valuable available asset")

df["Concurrent Credits"] = pd.to_numeric(df["Concurrent Credits"])
df.hist(column="Concurrent Credits")

df["Type of apartment"] = pd.to_numeric(df["Type of apartment"])
df.hist(column="Type of apartment")

df["No of Credits at this Bank"] = pd.to_numeric(df["No of Credits at this Bank"])
df.hist(column="No of Credits at this Bank")

df["Occupation"] = pd.to_numeric(df["Occupation"])
df.hist(column="Occupation")

df["No of dependents"] = pd.to_numeric(df["No of dependents"])
df.hist(column="No of dependents")

df["Telephone"] = pd.to_numeric(df["Telephone"])
df.hist(column="Telephone")

df["Foreign Worker"] = pd.to_numeric(df["Foreign Worker"])
df.hist(column="Foreign Worker")

plt.figure(figsize = (18,7))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)

df.describe()

#setting data back to normal
df=pd.DataFrame(data[0])
for col in df.columns:
  if df[col].dtype=='object':
    df[col] = df[col].str.decode('utf-8')
print(df.dtypes)

#creating filtered set of data containing 15/20 variables
newdata = df.drop(columns=["Foreign Worker", "No of dependents", "Duration in Current address", "Guarantors", "Telephone"])
newdata = pd.DataFrame(newdata)
print(newdata)

##Decision Tree on Unfiltered Data

# First split the data into train and test set
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
# Our class column is Creditability here and everything else will be used as features 
class_col_name='Creditability' 

feature_names=df.columns[df.columns != class_col_name ]
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, feature_names], df[class_col_name], test_size=0.3,random_state=1) 

from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)
print("Successfuly trained the decision tree...")

import graphviz
#Get unique class values to display on the tree
class_values=df[class_col_name].unique()
print ("class Names",class_values)


dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=feature_names,  
                                class_names=class_values,
                                filled=True)
# Plot tree
graph = graphviz.Source(dot_data, format="png") 
graph

# Let's make the prdictions on the test set that we set aside earlier using the trained tree
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test, y_pred)
print ("Confusion Matrix")
print(cf)
tn, fp, fn, tp=cf.ravel()
print ("TP: ", tp,", FP: ", fp,", TN: ", tn,", FN:", fn)

#print precision, recall, and accuracy from the perspective of each of the class (0 and 1 for German dataset)
from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))



##Decision Tree on Filtered Data

# First split the data into train and test set
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
# Our class column is Creditability here and everything else will be used as features 
class_col_name='Creditability' 

feature_names=newdata.columns[newdata.columns != class_col_name ]
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(newdata.loc[:, feature_names], newdata[class_col_name], test_size=0.3,random_state=1) 

from sklearn import tree
newclf = tree.DecisionTreeClassifier(max_depth=5)
newclf = newclf.fit(X_train, y_train)
print("Successfuly trained the decision tree...")

import graphviz
#Get unique class values to display on the tree
class_values=newdata[class_col_name].unique()
print ("class Names",class_values)


dot_data = tree.export_graphviz(newclf, out_file=None, 
                                feature_names=feature_names,  
                                class_names=class_values,
                                filled=True)
# Plot tree
graph = graphviz.Source(dot_data, format="png") 
graph

y_newpred = newclf.predict(X_test)

from sklearn.metrics import confusion_matrix
cf=confusion_matrix(y_test, y_newpred)
print ("Confusion Matrix")
print(cf)
tn, fp, fn, tp=cf.ravel()
print ("TP: ", tp,", FP: ", fp,", TN: ", tn,", FN:", fn)

#print precision, recall, and accuracy from the perspective of each of the class (0 and 1 for German dataset)
from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_newpred))


##Naive bayes classifier on unfiltered original dataset

from sklearn.naive_bayes import MultinomialNB

#Create a MultiNomial NB Classifier
nb = MultinomialNB()

#Train the model using the training sets
nb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = nb.predict(X_test)

print ("Total Columns (including class)",len(df.columns))

print("Number of features used ",nb.n_features_)
print("Classes ",nb.classes_)
print("Number of records for classes ",nb.class_count_)
print("Log prior probability for classes ", nb.class_log_prior_)
print("Log conditional probability for each feature given a class\n",nb.feature_log_prob_)

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))

##Naive bayes classifier on filtered original dataset

from sklearn.naive_bayes import MultinomialNB

#Create a MultiNomial NB Classifier
nb = MultinomialNB()

#Train the model using the training sets
nb.fit(X_train, y_train)

#Predict the response for test dataset
y_prednew = nb.predict(X_test)

print ("Total Columns (including class)",len(newdata.columns))

print("Number of features used ",nb.n_features_)
print("Classes ",nb.classes_)
print("Number of records for classes ",nb.class_count_)
print("Log prior probability for classes ", nb.class_log_prior_)
print("Log conditional probability for each feature given a class\n",nb.feature_log_prob_)

from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_newpred))