
# coding: utf-8

# In[37]:

import numpy as np


# In[38]:

import pandas as pd


# In[39]:

columns_names = ['user_id','item_id','rating','timestamp']


# In[40]:

df = pd.read_csv('u.data',sep = '\t',names = columns_names)


# In[41]:

df.head()


# In[42]:

movie_titles = pd.read_csv('Movie_Id_Titles')


# In[43]:

movie_titles.head()


# In[44]:

df = pd.merge(df,movie_titles, on = 'item_id')


# In[45]:

df.head()


# In[46]:

import matplotlib.pyplot as plt


# In[47]:

import seaborn as sns


# In[48]:

sns.set_style('white')


# In[49]:

get_ipython().magic('matplotlib inline')


# In[50]:

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[51]:

df.groupby('title')['rating'].count().sort_values(ascending = False).head()


# In[52]:

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[53]:

ratings.head()


# In[54]:

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[55]:

ratings.head()


# In[56]:

ratings['num of ratings'].hist(bins = 70)


# In[57]:

ratings['rating'].hist(bins = 70)


# In[58]:

sns.jointplot(x = 'rating',y = 'num of ratings', data = ratings,alpha = 0.5)


# In[59]:

moviemat = df.pivot_table(index = 'user_id',columns = 'title',values = 'rating')


# In[60]:

moviemat.head()


# In[61]:

ratings.sort_values('num of ratings', ascending = False).head(10)


# In[64]:

starwars_user_ratings = moviemat['Star Wars (1977)']
liar_liar_user_ratings = moviemat['Liar Liar (1997)'] 


# In[65]:

starwars_user_ratings.head()


# In[67]:

similiar_to_starwars = moviemat.corrwith(starwars_user_ratings)


# In[68]:

similiar_to_liarliar = moviemat.corrwith(starwars_user_ratings)


# In[69]:

corr_starwars = pd.DataFrame(similiar_to_starwars,columns = ['Correlation'])
corr_starwars.dropna(inplace=True)


# In[70]:

corr_starwars.head()


# In[71]:

corr_starwars.sort_values('Correlation', ascending = False ).head(10)


# In[72]:

corr_starwars = corr_starwars.join(ratings['num of ratings'])


# In[73]:

corr_starwars.head()


# In[75]:

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending = False).head()


# In[76]:

corr_liarliar = pd.DataFrame(similiar_to_liarliar,columns=['Correlation'])


# In[78]:

corr_liarliar.dropna(inplace=True)


# In[79]:

corr_liarliar = corr_liarliar.join(ratings['num of ratings'])


# In[88]:

corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:



