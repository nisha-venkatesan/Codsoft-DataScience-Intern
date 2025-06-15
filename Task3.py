#!/usr/bin/env python
# coding: utf-8

# ### TASK - 3 : IRIS Flower Classification
# 
# Domain : Data Science
# 
# Batch : June-25

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


task3 = pd.read_csv ("C:/Users/nisha/Downloads/IRIS.csv")
task3.head(11)


# In[9]:


task3.tail()


# In[10]:


task3['species'].value_counts()


# In[11]:


task3['species'].value_counts()


# In[12]:


sns.pairplot(task3 , hue = 'species')


# In[13]:


x = task3[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = task3["species"]


# In[14]:


print(x)
print(y)


# In[15]:


x_train , x_test, y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state=0)
x_train


# In[16]:


x_test


# In[18]:


from sklearn.neighbors import KNeighborsClassifier

ML = KNeighborsClassifier(n_neighbors=10)
ML.fit(x_train, y_train)


# In[19]:


x_train_prediction = ML.predict(x_train)
print(x_train_prediction)


# In[21]:


x_test_prediction = ML.predict(x_test)
x_test_prediction


# In[20]:


print (y_train)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train the model
ML = KNeighborsClassifier(n_neighbors=10)
ML.fit(x_train, y_train)

# Predictions
x_train_prediction = ML.predict(x_train)
x_test_prediction = ML.predict(x_test)

# Accuracy scores
train_accuracy = accuracy_score(y_train, x_train_prediction)
test_accuracy = accuracy_score(y_test, x_test_prediction)

print("Accuracy scores of training and test data are", train_accuracy, "and", test_accuracy, "respectively")


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
ML = KNeighborsClassifier(n_neighbors=10)
ML.fit(x_train, y_train)

# Predictions
x_train_prediction = ML.predict(x_train)
x_test_prediction = ML.predict(x_test)

# Accuracy
train_accuracy = accuracy_score(y_train, x_train_prediction)
test_accuracy = accuracy_score(y_test, x_test_prediction)
print("Accuracy scores of training and test data are", train_accuracy, "and", test_accuracy, "respectively")

# Classification report
print("Classification Report for Training Data:")
print(classification_report(y_train, x_train_prediction))

print("Classification Report for Test Data:")
print(classification_report(y_test, x_test_prediction))


# In[28]:


print ( " write sepal length" )
a = input()
print ( " write sepal width" )
b = input()
print ( " write petal length" )
c = input()
print ( " write sepal width" )
d = input()

data = pd.DataFrame({"sepal_length" : [a] , "sepal_width" : [b] , "petal_length" : [c], "petal_width": [d] })
prediction = ML.predict(data)
print ( "species is " , prediction[0] )


# In[ ]:




