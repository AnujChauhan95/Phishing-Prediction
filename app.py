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
