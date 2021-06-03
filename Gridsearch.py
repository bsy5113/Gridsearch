# Homework 2 - Bo Yun
#%%
# Package import --------------------------------------------------------------------
#####################################################################################

# Importing different metrics packages
import itertools
from typing import final
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Importing different ML algorithms 
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn.svm import SVC # SVM

# Kfold
from sklearn.model_selection import KFold

# Matplotlib
from matplotlib import pyplot as plt

# Product
from itertools import product 


#%%
# QUESTION 1. Function to take a list or dictionary of clfs and hypers --------------
#####################################################################################
import itertools

# Function 1. Using itertools product to get all possible combinations of hyperparameters
def OutputParamCombo(InputParamSet): 
    final_combinations={}
    for i in InputParamSet:
        SecondCombo=[]
        ParamOnly=list(clf_hyper[i].values())
        FirstCombo=list(itertools.product(*ParamOnly)) 
        for j in FirstCombo:
            SecondCombo.append(dict(zip(list(InputParamSet[i].keys()),j)))
        final_combinations[i]=SecondCombo
    return final_combinations

# Function 2. Calculating Kfold number of each combination model and averaging the scoring metrics
def KfoldAverageScore(model,metrics):
    M, L, n_folds = data
    kf = KFold(n_splits=n_folds)

    # Initializing dictionary with each score metrics
    scores = {}
    
    #init scores holder
    for i in metrics:
        scores[i.__name__] = []

    for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
        model.fit(M[train_index], L[train_index])
        pred = model.predict(M[test_index])
        for i in metrics:
            score = i(L[test_index], pred)
            scores[i.__name__].append(score)

    # Averaging nfolds of each score metric
    for k in scores:
        scores[k] = np.average(scores[k])
    
    return scores


# Function 3. Combining all functions together
def run(a_clf, data, clf_hyper={}):
    M, L, n_folds = data
    SUMMARY=[]
    metrics = [accuracy_score,recall_score, precision_score]
    # Creating all possible models with different combinations of hyperparameters
    FinalParamCombo=OutputParamCombo(clf_hyper)
    for i in a_clf:
        for j in FinalParamCombo[i.__name__]:
            model = i(**j)
            finalscore = KfoldAverageScore(model,metrics)
            SUMMARY.append({'clf': model,'score': finalscore})
    return SUMMARY



#%%
# QUESTION 2. Setting up for multiple classifiers and parameters --------------------
#####################################################################################

clfs=[LogisticRegression, KNeighborsClassifier,RandomForestClassifier, SVC]

clf_hyper={
    'LogisticRegression':{
        "solver":["newton-cg", "lbfgs", "liblinear"],
        "tol":[0.1,0.01,0.0001],
        "C":[5,10,15],
        },
    'KNeighborsClassifier':{
        "n_neighbors":[1,3,5],
        "p":[1,2,3],
        "leaf_size":[1,3,5],
        },
    'RandomForestClassifier':{
        "n_jobs":[1,3,5],
        "max_depth":[5,10,15],
        "max_features":["auto", "sqrt", "log2"]
        },
    'SVC':{
        "C":[1,3,5],
        "gamma":[5,10,15],
        "kernel":["linear","rbf","sigmoid"]}}

#%%
# QUESTION 3. Data------------------------------------------------------------------
#####################################################################################

import numpy as np
import pandas as pd
import csv
from sklearn import datasets

df = datasets.load_breast_cancer()
M=df['data']
L=df['target']

n_folds = 3
kf = KFold(n_splits=n_folds)
# Data compile 
data = (M, L, n_folds)
KFold(n_splits=n_folds)

# Showing what kf.split is doing
for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
	print("k fold = ", ids)
	print("train indexes", train_index)
	print("test indexes", test_index)


#%%
# QUESTION 4. Execution of functions and matplotlib plot ----------------------------
#####################################################################################

# Putting it all together
FINAL=run(clfs,data,clf_hyper=clf_hyper)

# Finding classifier model with the maximum accuracy score 
LR=[]
KNN=[]
RF=[]
SVM=[]

for i in FINAL:
    if str(i['clf']).split('(')[0]=='LogisticRegression':
        LR.append(i['score']['accuracy_score'])
    if str(i['clf']).split('(')[0]=='KNeighborsClassifier':
        KNN.append(i['score']['accuracy_score'])
    if str(i['clf']).split('(')[0]=='RandomForestClassifier':
        RF.append(i['score']['accuracy_score'])
    if str(i['clf']).split('(')[0]=='SVC':
        SVM.append(i['score']['accuracy_score'])


max_LR_index = LR.index(max(LR))
max_KNN_index = KNN.index(max(KNN))
max_RF_index = RF.index(max(RF))
max_SVM_index = SVM.index(max(SVM))

print("The LR model with the highest accuracy is:", FINAL[max_LR_index])
print("The KNN model with the highest accuracy is:", FINAL[len(LR)-1+max_KNN_index])
print("The RF model with the highest accuracy is:", FINAL[len(LR)+len(KNN)-2+max_RF_index])
print("The SVM model with the highest accuracy is:", FINAL[len(LR)+len(KNN)+len(RF)-3+max_SVM_index])

# Best model of each classifier
Best_Model_LR = FINAL[max_LR_index]
Best_Model_KNN = FINAL[len(LR)-1+max_KNN_index]
Best_Model_RF = FINAL[len(LR)+len(KNN)-2+max_RF_index]
Best_Model_SVM = FINAL[len(LR)+len(KNN)+len(RF)-3+max_SVM_index]


# Boxplot comparison of each classifier
nm=['LR','KNN','RF','SVM']
x=[1,2,3,4]
plt.title('Classification result')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.boxplot([LR,KNN,RF,SVM])
plt.xticks(x,nm)
plt.show()

"""
Boxplot result:

RandomForest and LogisticRegression are my top choice as they don't show much variance throughout kfolds and the accuracies are high.
On the other hand, Support vector machine is not good as it shows a huge variance among different kfold runs. 

"""
#%%
# QUESTION 5. Saving variables ------------------------------------------------------
#####################################################################################

# Saving variables using pickle
import pickle

# Important variables to save
input_variables=[Best_Model_KNN, Best_Model_LR,Best_Model_RF,Best_Model_SVM,LR,KNN,RF,SVM]

# Saving the objects:
with open('objs.pkl', 'wb') as f:  
    pickle.dump(input_variables, f)

# Open variables using the following
with open('objs.pkl','rb') as f:  
    input_variables = pickle.load(f)

# # Saving the boxplot image
# image=user/boyun/Desktop/Clf_result.png
# file = open('BoxplotImage.pkl', 'wb')
# pickle.dump(image, file)
# file.close()

# # Opening up image
# file = open('BoxplotImage.pkl', 'rb')
# image = pickle.load(file)
# print(image)
# file.close()


#%%
# QUESTION 6. Gridsearch ------------------------------------------------------------
#####################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

# Splitting Train and test dataset

models={'LogisticRegression': LogisticRegression(), 'KNeighborsClassifier':KNeighborsClassifier(),'RandomForestClassifier':RandomForestClassifier(), 'SVC':SVC()}
# models=[('LogisticRegression',LogisticRegression()), ('KNeighborsClassifier',KNeighborsClassifier()),('RandomForestClassifier',RandomForestClassifier()), ('SVC',SVC())]

def grid(data,target):
    for k in models.keys():
        model=models[k]
        param=clf_hyper[k]
        gridsearch=GridSearchCV(estimator=model,param_grid=param, scoring='accuracy')
        gridsearch.fit(data,target)
   
        # print classification report 
        print("Best parameters are: {}".format(gridsearch.best_estimator_))
        print ("Accuracy score is {}".format(gridsearch.best_score_))

# Execution of function
grid(M,L)

"""
*** Gridsearch result discussion

[Logistic Regression]
Gridsearch: Best parameters are: LogisticRegression(C=15, solver='liblinear'), Accuracy score is 0.9560782487191428
MyModel: {'clf': LogisticRegression(C=10, solver='liblinear'), 'score': {'accuracy_score': 0.9560568086883876, 'recall_score': 0.9747843179353773, 'precision_score': 0.9541887677056637}}

[KNN]
Gridsearch: Best parameters are: KNeighborsClassifier(leaf_size=1, p=1), Accuracy score is 0.9314547430523211
MyModel: {'clf': KNeighborsClassifier(leaf_size=5, n_neighbors=3, p=3), 'score': {'accuracy_score': 0.9103685138772857, 'recall_score': 0.9431595477230116, 'precision_score': 0.91504417261298}}


[RF]
Gridsearch:Best parameters are: RandomForestClassifier(max_depth=15, n_jobs=3), Accuracy score is 0.9666356155876417
MyModel:{'clf': RandomForestClassifier(max_depth=5, n_jobs=3), 'score': {'accuracy_score': 0.9578390420495685, 'recall_score': 0.9750069279567396, 'precision_score': 0.9530156184632549}}

[SVM]
Gridsearch: Best parameters are: SVC(C=3, gamma=5, kernel='linear'), Accuracy score is 0.9508306163639186
MyModel:{'clf': SVC(C=1, gamma=15, kernel='linear'), 'score': {'accuracy_score': 0.945521210433491, 'recall_score': 0.9702181078897153, 'precision_score': 0.9425405951097651}}

=> Overall, there are some differences in picking parameters but the accuracy relatively stayed the similar. 
"""



