#!/usr/bin/env python
# coding: utf-8

# In[389]:


import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\ADMIN\Desktop\Usha\dataset\Tweets.csv")
df.head()


# In[390]:


import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
df = df.reindex(np.random.permutation(df.index))
df['airline_sentiment'][0]
df['text'][0]
copy=df['airline_sentiment'],df['text']
print(copy)


# In[391]:


review=review.lower()
review=review.split()
ps=PorterStemmer()
review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review= ' '.join(review)
c.append(review)


# In[392]:


from sklearn.feature_extraction.text import CountVectorizer  
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(c).toarray()
with open('CountVectorizer','wb') as file:
    pickle.dump(cv,file)
y=df.iloc[:,1].values


# In[393]:


from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from keras import layers
from keras import regularizers
from keras.models import load_model
import pickle


# In[394]:


X_train_rest, X_test, y_train_rest, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=37)


# In[395]:


assert X_test.shape[0] == y_test.shape[0]
assert X_train_rest.shape[0] == y_train_rest.shape[0]
test_set = df[:1000]


# In[396]:


print('Shape of validation set:',X_test.shape)

base_model = models.Sequential()
base_model.add(layers.Dense(units=64,kernel_initializer='uniform', activation='relu' ))
base_model.add(layers.Dense(units=64,kernel_initializer='uniform', activation='relu'))
base_model.add(layers.Dense(units=3,kernel_initializer='uniform', activation='softmax'))
X_test.shape


# In[397]:


def deep_model(base_model):
    base_model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
    base_model.fit(X_train, y_train, epochs=7, batch_size=51,  verbose=0)


# In[398]:


model.save('mymodel.h5')


# In[399]:


pred_df = pd.DataFrame({'airline_sentiment':test_set['airline_sentiment'],'text':test_set['text']})


# In[400]:


pred_df['airline_sentiment'] = pred_df['airline_sentiment'].map({0: 'Negative', 1: 'Neutral', 2 : 'Positive'})
pred_df['text'] = pred_df['text'].map({0: 'Negative', 1: 'Neutral', 2 : 'Positive'})


# In[401]:


pred_df['pred agreement'] = (pred_df['airline_sentiment'] == pred_df['text'])
print(f"The models agree with each other {round(pred_df['pred agreement'].value_counts()/len(pred_df)*100, 4)}% of the time.")


# In[402]:


pred_df.to_csv(r'Tweets.csv', index=False)


# In[403]:


df_test = pd.read_csv('Tweets.csv'); df_test.head(4)




# In[325]:


model.save('mymodel.h5')


# In[ ]:




