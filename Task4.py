#!/usr/bin/env python
# coding: utf-8

# ### Task-4 [SALES PREDICTION USING PYTHON]
# 
# Domain:Data Science
# 
# Batch : June-25

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/nisha/Downloads/advertising (1).csv")
df.head(10)


# In[3]:


df.shape


# In[4]:


df.describe


# In[5]:


sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# In[6]:


df['TV'].plot.hist()


# In[26]:


df['Radio'].plot.hist(color='Black')


# In[25]:


df['Newspaper'].plot.hist(color='red')


# In[9]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[11]:


from sklearn.model_selection import train_test_split

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size=0.3, random_state=0)

# Display training features
print(x_train)


# In[12]:


print (y_train)


# In[14]:


print (x_test)


# In[15]:


print (y_test)


# In[16]:


print (y_test)


# In[18]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)


# In[19]:


res = model.predict(x_test)
print(res)


# In[20]:


model.coef_


# In[22]:


model.intercept_


# In[23]:


plt.plot(res)


# In[24]:


plt.scatter(x_test,y_test)
plt.plot(x_test,res, color='red')
plt.show()


# In[ ]:




