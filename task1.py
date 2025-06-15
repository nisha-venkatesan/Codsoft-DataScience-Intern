#!/usr/bin/env python
# coding: utf-8

# ### Task1 [TITANIC SURVIVAL PREDICTION]
# 
# DOMAIN : Data Science
# 
# Batch: June-25

# In[2]:


import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


titanic = pd.read_csv("C:/Users/nisha/Downloads/Titanic-Dataset (1).csv")


# In[4]:


titanic.shape


# In[5]:


intern = pd.read_csv("C:/Users/nisha/Downloads/Titanic-Dataset (1).csv")
intern.shape


# In[6]:


titanic.head(10)


# In[8]:


titanic.describe()


# In[12]:


print(titanic.columns)


# In[13]:


titanic["Sex"]


# In[14]:


titanic['Survived'].value_counts()


# In[15]:


sns.countplot(x=titanic["Survived"], hue=titanic['Pclass'])


# In[16]:


sns.countplot(x=titanic["Sex"], hue=titanic['Survived'])


# In[17]:


titanic['Sex'].unique()


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
titanic['Sex'] =le.fit_transform(titanic['Sex'])
titanic.head()


# In[19]:


titanic['Sex'] , titanic['Survived']


# In[20]:


sns.countplot(x=titanic["Sex"], hue=titanic['Survived'])


# In[21]:


titanic.isna().sum()


# In[22]:


intern = titanic.drop(['Age'], axis =1)
intern_final = titanic
intern_final.head()


# In[23]:


from matplotlib import pyplot as plt
intern_final['Survived'].plot(kind='line', figsize=(8, 4), title='Survived')
plt.gca().spines[['top', 'right']].set_visible(False)


# In[25]:


log.fit(x_train, Y_train)


# In[26]:


prediction = print(log.predict(x_test))


# In[27]:


print(Y_test)


# In[29]:


import warnings
warnings.filterwarnings('ignore')

res = log.predict([[1, 1]])

if res == 0:
    print('Not Survived')
else:
    print('Survived')


# In[ ]:




