#!/usr/bin/env python
# coding: utf-8

# In[9]:

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


df=pd.read_csv(r'C:\Users\USER\Desktop\cyber security\phishing\web-page-phishing.csv')


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


cat_col = ['n_at','n_tilde','n_redirection']
for i in cat_col:
    print(i)
    df[i] = df[i].fillna(df[i].median())


# In[14]:


from sklearn.metrics import accuracy_score, classification_report,precision_score


# In[15]:


X = df.loc[:, ['url_length', 'n_dots', 'n_hypens', 'n_underline', 'n_slash',
               'n_questionmark', 'n_redirection']]
Y = df['phishing']


# In[27]:


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# In[29]:


params = {
    "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
}
grcv= RandomizedSearchCV(XGBClassifier(random_state=32), params, n_jobs=-1, cv=3)
grcv.fit(X, Y)
grcv.best_params_


# In[35]:


from sklearn.model_selection import train_test_split


# In[37]:


xtrain, xtest, ytrain , ytest = train_test_split(X,Y, random_state=67,test_size=0.20, stratify=Y)


# In[39]:


xgb =  XGBClassifier(max_depth=12, min_child_weight= 3,learning_rate=0.15,gamma=0.3,colsample_bytree= 0.5)
xgb.fit(xtrain, ytrain)
xgb_pred =xgb.predict(xtest)


# In[43]:


print("Accuracy:", accuracy_score(ytest, xgb_pred))

print("Classification Report:\n", classification_report(ytest, xgb_pred))


# In[45]:


import joblib


# In[47]:


joblib.dump(xgb, 'model.pkl')


# In[ ]:




