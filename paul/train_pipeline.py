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

#retreat date type such as date
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