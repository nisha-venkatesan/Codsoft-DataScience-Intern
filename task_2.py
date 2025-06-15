#!/usr/bin/env python
# coding: utf-8

# ### MOVIE RATING PREDICTION WITH PYTHON
Batch = june 2025
Domain = Data Science
# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df =pd.read_csv(("C:/Users/nisha/OneDrive/Desktop/IMDb Movies India.csv"),encoding='unicode_escape')
df.head(11)


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.info()


# In[7]:


df.describe(include='all')


# In[8]:


df.duplicated().sum()


# In[9]:


df.shape


# In[10]:


df.dropna(inplace=True)
df.isnull().sum()


# In[11]:


df.drop_duplicates(inplace=True)
df . shape


# In[12]:


df.info()


# In[14]:


df.describe()


# In[17]:


df.columns


# In[18]:


df['Year' ] = df['Year'].fillna(0)
df['Year' ] = df['Year'].replace(r'[()]', '', regex=True).astype(int)
print(df['Year'])


# In[19]:


df['Duration'] = pd.to_numeric(df['Duration'].str.replace(' min', ''))
genres = df['Genre'].value_counts()
genres


# In[32]:


import numpy as np
import matplotlib.pyplot as plt

df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre')
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])  # ← Fixed here

genres = df['Genre'].value_counts()
top_genres = genres.head(10)

plt.figure(figsize=(8, 8))
colors = plt.cm.viridis(np.linspace(0.7, 0.9, len(top_genres)))
plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%', colors=colors)
plt.title('Top 10 Genres with Total Number of Movies')
plt.show()


# In[21]:


Year = df['Year'].value_counts()
Year


# In[22]:


top_Year = Year.head(10)
plt.figure(figsize=(8,8))
colors = plt.cm.viridis(np.linspace(0.7, 0.9, len(top_Year)))
plt.pie(top_Year.values, labels=top_Year.index, autopct='%1.1f%%', colors=colors)
plt.title('Top 10 year with Total Number of Movies')
plt.show()


# In[23]:


directors = df['Director'].value_counts()
directors


# In[24]:


top_directors = directors.head(10)
plt.figure(figsize=(8, 8))
colors = plt.cm.viridis(np.linspace(0.7, 0.9, len(top_directors)))
plt.pie(top_directors.values, labels=top_directors. index, autopct='%1.1f%%', colors=colors)
plt.title('Top 10 directors with Total Number of Movies')
plt.show()


# In[25]:


actors = pd.concat([df['Actor 1'], df['Actor 2'], df['Actor 3']]).value_counts()
actors


# In[26]:


Top_actors = actors.head(10)


1

1

plt.figure(figsize=(8, 8))
colors = plt.cm.viridis(np.linspace(0.7, 0.9, len(Top_actors)))
plt.pie(Top_actors.values, labels=Top_actors.index, autopct='%1.1f%%', colors=colors)
plt.title('Top 10 actors with Total Number of Movies')
plt.show()


# In[30]:


df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre')
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])
print(df.head(10))



# In[31]:


def clean_duration(duration):
    if isinstance(duration, str):
        return float(''.join(filter(str.isdigit, duration)))
    return duration

df['Duration'] = df['Duration'].apply(clean_duration)

df['Votes'] = df['Votes'].astype(str)
df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

df['Year'] = df['Year'].astype(str)
df['Year'] = df['Year'].str.strip('()').astype(int)

df.info()
df


# In[33]:


df = df.drop(columns=['Name'])
actor1_encoding_map = df.groupby('Actor 1').agg({'Rating': 'mean'}).to_dict()
actor2_encoding_map = df.groupby('Actor 2').agg({'Rating': 'mean'}).to_dict()
actor3_encoding_map = df.groupby('Actor 3').agg({'Rating': 'mean'}).to_dict()
director_encoding_map = df.groupby('Director').agg({'Rating': 'mean'}).to_dict()
genre_encoding_map = df.groupby('Genre').agg({'Rating': 'mean'}).to_dict()

df['encoded_actor1'] = round(df['Actor 1'].map(actor1_encoding_map['Rating']),1)
df['encoded_actor2'] = round(df['Actor 2'].map(actor2_encoding_map['Rating']),1)
df['encoded_actor3'] = round(df['Actor 3'].map(actor3_encoding_map['Rating']),1)
df['encoded_director'] = round(df['Director'].map(director_encoding_map['Rating']),1)
df['encoded_genre'] = round(df['Genre'].map(genre_encoding_map['Rating']),1)

df.drop(['Actor 1', 'Actor 2', 'Actor 3', 'Director', 'Genre'], axis=1, inplace=True)
df


# In[34]:


test_data = df.drop(columns=['Rating'])
test_data


# In[35]:


ratings = df['Rating']
ratings


# In[ ]:




