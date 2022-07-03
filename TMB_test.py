# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 22:25:45 2022

@author: khanh
"""
import pandas as pd
import pickle

filename = 'models/finalized_LGBM_model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

df_tst = pd.read_csv('data/TMB.trn.LGBM-GA.csv')
X_tst = df_tst.drop('TMB_group', axis=1)
y_tst = df_tst.TMB_group
result = loaded_model.score(X_tst, y_tst)
print(result)