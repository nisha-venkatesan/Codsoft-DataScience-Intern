#!/usr/bin/env python
# coding: utf-8

# ### CREDIT CARD FRAUD DETECTION
# 
# Domain: Data Science
# 
# Batch: June-25

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import warnings


# In[2]:


intern_df = pd.read_csv("C:/Users/nisha/OneDrive/Desktop/creditcard.csv")


# In[3]:


intern_df = intern_df.sample(n=50000, random_state=1)


# In[4]:


intern_df.head(10)


# In[5]:


intern_df.shape


# In[6]:


intern_df.columns


# In[7]:


intern_df.isnull()


# In[8]:


intern_df.isnull().sum()


# In[10]:


intern_df.describe()


# In[14]:


intern_df['Class'].value_counts()


# In[12]:


intern_df.info()


# In[15]:


fraud=intern_df[intern_df['Class'] == 1]
genuine=intern_df[intern_df['Class'] == 0]
fraud.Amount.describe()


# In[16]:


genuine.Amount.describe()


# In[20]:


intern_df.hist(figsize=(20,20),color='Black')
plt.show()


# In[18]:


rcParams['figure.figsize']= 16, 8
f,(ax1,ax2)=plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(fraud.Time, fraud.Amount)
ax1.set_title('Fraud')
ax2.scatter(genuine.Time, genuine.Amount)
ax2.set_title('Genuine')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
plt.show()


# In[ ]:




