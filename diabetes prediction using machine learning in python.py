#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_csv('diabetes.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.dtypes


# In[7]:


data.describe()


# In[8]:


data.isna().sum()


# In[11]:


data=data.dropna(axis=1)


# In[12]:


data.info


# In[13]:


data['Outcome'].value_counts()


# In[14]:


# to calculate the mean of dataset
data.groupby('Outcome').mean()


# In[15]:


#separating the data into independent and dependent variable 
x= data.drop(columns = 'Outcome', axis=1)
y=data['Outcome']


# In[16]:


print(x)


# In[17]:


print(y)


# In[19]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[20]:


#data standardization
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)


# In[22]:


x_train


# In[24]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# In[25]:


#predicting the test set result
y_pred=classifier.predict(x_test)


# In[27]:


#testing the accuracy of the result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[ ]:




