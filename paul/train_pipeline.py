#!/usr/bin/env python
# -*- coding: utf-8 -*-

##------------------------------------------------------------##
#Imports
import math
import json
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import *
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

##------------------------------------------------------------##
#load data
df = pd.read_excel('../data/synthese.xlsx')
aux = pd.read_excel('../data/extrait_2.xlsx')

#drop useless
to_drop = ["annee","GME_ATIH","chapitre","GN","RGME","sortie",u'Médecin',"dep_totale"]
target = ["nb_jours_semaine"]
target_related = [ u'nb_rhs', u'nb_jours_total', u'nb_jours_WE']
rename_mapper = {"date_2":"date_entree",
                 "jour":"jour_entree",
                 "mois":"mois_entree",
                 "annee":"annee_entree"}
df = df.drop(to_drop,axis=1)
aux= aux.drop(["date"], axis = 1).rename(rename_mapper, axis=1)

#cast date type such as date
aux["date_entree"]= pd.to_datetime(aux["date_entree"])

#left join on N_ordre
data = pd.merge(df,aux,on="N_ordre",how="left")

#fill na to 0 for the 
practiciens = [u'Animateur', u'Assistant de service social',
                 u'Autre intervenant',                 u'Diététicien',
              u'Éducateur spécialisé',           u'Éducateur sportif',
                u'Enseignant général',                    u'Ergonome',
                    u'Ergothérapeute',                   u'Infirmier',
           u'Masseurkinésithérapeute',        u'Moniteur d’autoécole',
                u'Moniteur éducateur',               u'Orthophoniste',
                  u'Orthoprothésiste',                  u'Osteopathe',
                       u'Psychologue',             u'Psychomotricien']

for col in practiciens:
    data[col] = data[col].fillna(0)
    
##------------------------------------------------------------##
#Features
target = ["nb_jours_semaine"]
#create age instead at time of entry
data["Age"] = data["annee_entree"] - data["annee_naissance"]

#create indicator for u'nb_jours_WE'
def weekend_indicator(x):
    """input : x scalar"""
    if x < 0.8:
        return 0
    if x>=0.8 and x<1.8:
        return 1
    if x>=1.8:
        return 2
data["weekend_indicator"] = (data["nb_jours_WE"]/data["nb_rhs"]).apply(lambda x :weekend_indicator(x))

#add nombre d'intervenants
def nb_inter(row,practiciens):
    """
    Input:
    row : row of dataframe
    practiciens : métier
    """
    count = 0
    for col in practiciens:
        if row[col]>0:
            count+=1
    return count
data["Nb_intervenants"] = data.apply(lambda row : nb_inter(row,practiciens),axis=1)

data['Total_interventions'] = data[practiciens].sum(axis=1)

#eliminate those who have Nb_intervenants = 0
data = data[data["Nb_intervenants"]>0].reset_index(drop = True)

#regroup by postal county
#CP est un type entier reformater en string avec 0 devant
data['CP'] = data['CP'].apply(lambda x : str(x).zfill(5))
data['CP_departement']=data['CP'].apply(lambda x : x[0:2])

#threshold and discretize 

#Creation de la variable “chapitre” simplifiee : 
#Correlation la plus grande avec “NbSemaines”
def simple_chap(chapitre):
    if "08" in chapitre:
        return 0
    if "01" in chapitre:
        return 1
    else:
        return 2
data["T_chapitre_discretized"] = data["T_chapitre"].apply(lambda x : simple_chap(x))


thresh_practiciens = {
     u'Animateur' : 909,
     u'Assistant de service social':243,
     u'Autre intervenant':227,
     u'Di\xe9t\xe9ticien':81,
     u'\xc9ducateur sp\xe9cialis\xe9':913,
     u'\xc9ducateur sportif':781,
     u'Enseignant g\xe9n\xe9ral':1888,
     u'Ergonome':258,
     u'Ergoth\xe9rapeute':781,
     u'Infirmier':411,
     u'Masseurkin\xe9sith\xe9rapeute':1419,
     u'Moniteur d\u2019auto\xe9cole':338,
     u'Moniteur \xe9ducateur':1715,
     u'Orthophoniste':945,
     u'Orthoproth\xe9siste':84,
     u'Osteopathe':194,
     u'Psychologue':364,
     u'Psychomotricien':577
}

#Discretise for each practicien:
for col in practiciens:
    thresh = thresh_practiciens[col]
    data[u"%s_discretized"%col]  = data[col].apply(lambda x : 1 if x>0 and x<=thresh else (2 if x>thresh else 0))
    
 #mean et mediane
GN_Avg_Med = pd.read_csv("../data/aux/GN_Avg_Med.csv",sep=",",encoding="utf8")
data = pd.merge(data,GN_Avg_Med,on="T_GN",how="left")
left_cols = [x+"_x" for x in GN_Avg_Med.columns if x != "T_GN"]

Chapitre_Avg_Med = pd.read_csv("../data/aux/Chapitre_Avg_Med.csv",sep=",",encoding="utf8").rename({"T_Chapitre":
                                                                                                  "T_chapitre"},axis=1)
data = pd.merge(data,Chapitre_Avg_Med,on="T_chapitre",how="left")
left_cols = left_cols + [x+"_y" for x in Chapitre_Avg_Med.columns if x != "T_chapitre"]

#set columns
seless = []+practiciens
option =["dep_sup","dep_physique","CP","jour_entree","mois_entree",
        "annee_entree","weekend_indicator","T_GN"]
keep = ["N_ordre","type hosp","nb_jours_semaine",
        "T_chapitre_discretized","sexe","Age","weekend_indicator",
        "entree","Nb_intervenants","dep_sup","dep_physique"]+left_cols

keep_ = []
for col in practiciens:
    keep_.append(u"%s_discretized"%col)
    
data = data[keep+keep_]
list_categorical = ["T_chapitre_discretized","sexe","entree"]
data = pd.get_dummies(data, columns=list_categorical,drop_first=True)

cols_dummies = [x for x in data.columns if any(y in x for y in list_categorical) ]
keep = [x for x in keep if x not in list_categorical + target+["N_ordre"]]+cols_dummies

##------------------------------------------------------------##
#ML part

from sklearn.metrics import mean_squared_error
from math import sqrt

# RMSE :root-mean-square error
def RMSE(y_true, y_pred): 
    return sqrt(mean_squared_error(y_true, y_pred))


# Mean Absolute Percentage Error
def mape_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Mean Absolute  Error
def mae_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred)))
  
import sklearn
from sklearn.cross_validation import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import grid_search
from sklearn.model_selection import GridSearchCV

# Sélection du model:

lr = LinearRegression() #init
polynomial_features = PolynomialFeatures(degree=2, include_bias=False)
X_ = polynomial_features.fit_transform(X[keep])
X_ = pd.DataFrame(X_)
X_ = pd.concat([X_,X[keep_]],axis=1)

X0_train_,X0_test_,y0_train_,y0_test_ = train_test_split(X_,\
                                                np.log(y),\
                                                train_size=26000,\
                                                random_state=42)

# Création du pipeline
pipe = Pipeline([
        ("linear_regression", lr)])#("polynomial_features", polynomial_features)
grid = dict(linear_regression__normalize=[False,True]) #espace de paramètre de la régression
                                                        # choix sur la normalisation
model = GridSearchCV(pipe,param_grid=grid,cv=8)
model.fit(X0_train_, y0_train_)
#predict
y_lr = model.predict(X0_test_)

##------------------------------------------------------------##
#Evaluation du modèle :#Evaluat 
print("L'erreur type (RMSE) est de %s") %(RMSE(y0_test_, y_lr))
print("La moyenne absolue d'erreur est de %s.") %(mae_error(y0_test_.values, y_lr))
print("La vrai moyenne absolue d'erreur est de %s.") %(mae_error(y0_test.values, np.exp(y_lr)))
print("Meilleur model utilisant %s." % ( model.best_params_))
print('Variance score: %.2f' % r2_score(y0_test_.values, y_lr))
print('Vrai Variance score: %.2f' % r2_score(y0_test.values, np.exp(y_lr)))
