#!/usr/bin/env python
# coding: utf-8

#  ## Amazon Reviews - Sentiment Analysis

# In[1]:


# import necessary packages
import pandas as pd
import numpy as np
import gzip
from string import punctuation

# natural language kit 
from nltk.corpus import stopwords
from nltk import word_tokenize

# graphing libraries
import matplotlib.pyplot as plt

# scikit learn libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, confusion_matrix


# save the model into python object structure
import pickle


#  ## Data Cleanup

# In[2]:


# parse json file into pandas dataframe
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('/Users/nithinsivakumar/projects/ubicomp/Dataset/reviews_Beauty_5.json.gz')


# In[3]:


# print first 5 rows
df.head()
df['reviewText'][0]


# In[4]:


# data cleaning
df['reviewText'] = df['reviewText'].str.lower()
df.head()


# In[5]:


# remove punctuations
def remove_punctuations(text):
    for punct in punctuation:
        text = text.replace(punct, ' ')
    return text


# In[6]:


# remove punctuations, stopwords
df['reviewText'] = df['reviewText'].apply(remove_punctuations)
# remove stopwords
stop = stopwords.words('english')
df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop]))
df.head()


# In[7]:


# creating a tokenized column and for the reviewText
df['reviewText_tok'] = df['reviewText'].apply(word_tokenize)

# creating a positive negative column where positive review (4-5) are 1 and negative reviews (1-3) are 0.
df['pos_neg'] = df['overall'].apply(lambda x: 1 if x > 3 else 0)
df.head()


# In[8]:


# print positve and negative reviews stats
no_tot_reviews = len(df.pos_neg)
no_pos_reviews = df.pos_neg.sum()
percent_pos_reviews = no_pos_reviews/no_tot_reviews*100
no_neg_reviews = no_tot_reviews - no_pos_reviews
percent_neg_revies = no_neg_reviews/no_tot_reviews*100

print('Number of overall reviews: ', no_tot_reviews)
print()

print('Number of positive reviews: ', no_pos_reviews)
print('Percent of Positve reviews: ',percent_pos_reviews)
print()

print('Number of negative reviews: ', no_neg_reviews)
print('Percent of reviews: ', percent_neg_revies)
print('Number of unique reviewers: ', len(set(df.reviewerID)))


# In[9]:


# Average Length of reviews
length = []
for review in df.reviewText_tok:
    length.append(len(review))
print('Average Length of Reviews: ', np.mean(length))


# In[10]:


max(length)


# In[11]:


plt.hist(length, bins=200);


# In[12]:


pos_reviews = df[df.pos_neg == 1]
pos_reviews.head()


# In[13]:


# Avg Length of Positve reviews
pos_length = []
for review in pos_reviews.reviewText_tok:
    pos_length.append(len(review))
print('Average Length of Positive Reviews: ', np.mean(pos_length))


# In[14]:


plt.hist(pos_length, bins=200);


# In[15]:


neg_reviews = df[df.pos_neg == 0]
neg_length = []
for review in neg_reviews.reviewText_tok:
    neg_length.append(len(review))
print('Average Length of Negative Reviews: ', np.mean(length))
plt.hist(neg_length, bins=200);


# ### Save Model 

# In[16]:


# save the model file using pickle
def saveModel(model, name):
    path = "../Models/" + name
    pickle.dump(model, open(path,"wb"))


# In[17]:


train_x, test_x, train_y, test_y = train_test_split(df.reviewText, df.pos_neg, random_state=42)


# # Naive Bayes

# In[18]:


print('Size of training set: ', len(train_x))
print('Size of test set: ', len(test_x))


# ### NB-Count Vectorizer

# In[19]:


CV = CountVectorizer()
x_train_CV = CV.fit_transform(train_x)
x_test_CV = CV.transform(test_x)


# In[20]:


NB_CV = MultinomialNB(alpha=1.6)
NB_CV.fit(x_train_CV, train_y)


# ### NB-Tfidf

# In[21]:


tfidf = TfidfVectorizer()
x_train_tf = tfidf.fit_transform(train_x)
x_test_tf = tfidf.transform(test_x)


# In[22]:


svd = TruncatedSVD()
svd.fit(x_train_tf)


# In[23]:


NB_tf = MultinomialNB(alpha=0.01)
NB_tf.fit(x_train_tf, train_y)


# ### Accuracy

# In[24]:


f1_tf = f1_score(test_y, NB_tf.predict(x_test_tf))
f1_CV = f1_score(test_y, NB_CV.predict(x_test_CV))
print('NB-CV ', f1_tf)
print('NB-TF ', f1_CV)


# #### Save model

# In[25]:


# serializing our model to a file called model.pkl
pickle.dump(NB_tf, open("../Models/model.pkl","wb"))
pickle.dump(NB_CV, open("../Models/model.pkl","wb"))


# # Logistic Regression

# ### Tfidf

# In[26]:


lr_tf = LogisticRegression()
lr_tf.fit(x_train_tf, train_y)


# ### Count Vectorizer

# In[27]:


lr_cv = LogisticRegression()
lr_cv.fit(x_train_CV, train_y)


# In[28]:


f1_lg_tf = lr_tf.score(x_test_tf, test_y)
f1_lg_cv = lr_cv.score(x_test_CV, test_y)
saveModel(lr_tf, "lr_tf.pkl")
saveModel(lr_cv, "lr_cv.pkl")


# In[29]:


print('LR-CV ', f1_lg_tf)
print('LR-TF ', f1_lg_cv)


# # Predict a single review

# In[46]:


# save train_x,y
with open("/Users/nithinsivakumar/projects/ubicomp/opinion-mining/Models/train.pkl", "wb") as f:
    pickle.dump([train_x, train_y], f)


# In[50]:


with open("/Users/nithinsivakumar/projects/ubicomp/opinion-mining/Models/train.pkl", "rb") as f:
    train_a, train_b = pickle.load(f)

cvN=CountVectorizer()
a_train_CV = cvN.fit_transform(train_a)
content = "texture concealer pallet fantastic great cover."
data = [content]
vect = cvN.transform(data).toarray()
model = pickle.load(open("/Users/nithinsivakumar/projects/ubicomp/opinion-mining/Models/model.pkl", "rb"))
my_prediction = model.predict(vect)
my_prediction[0]


# In[ ]:




