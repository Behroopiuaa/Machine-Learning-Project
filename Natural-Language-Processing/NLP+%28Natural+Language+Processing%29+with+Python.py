
# coding: utf-8

# 
# ___
# # NLP (Natural Language Processing) with Python
# 

# In[1]:

# ONLY RUN THIS CELL IF YOU NEED 
# TO DOWNLOAD NLTK AND HAVE CONDA
# WATCH THE VIDEO FOR FULL INSTRUCTIONS ON THIS STEP

# Uncomment the code below and run:


# !conda install nltk #This installs nltk
# import nltk # Imports the library
# nltk.download() #Download the necessary datasets


# ## Get the Data

# We'll be using a dataset from the [UCI datasets]

# The file we are using contains a collection of more than 5 thousand SMS phone messages.
# 
# Let's go ahead and use rstrip() plus a list comprehension to get a list of all the lines of text messages:

# In[3]:

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))


# A collection of texts is also sometimes called "corpus". Let's print the first ten messages and number them using **enumerate**:

# In[4]:

for message_no, message in enumerate(messages[:10]):
    print(message_no, message)
    print('\n')


# Due to the spacing we can tell that this is a [TSV]("tab separated values") file, where the first column is a label saying whether the given message is a normal message (commonly known as "ham") or "spam". The second column is the message itself. (Note our numbers aren't part of the file, they are just from the **enumerate** call).
# 
# Using these labeled ham and spam examples, we'll **train a machine learning model to learn to discriminate between ham/spam automatically**. Then, with a trained model, we'll be able to **classify arbitrary unlabeled messages** as ham or spam.
# 
# 

# Instead of parsing TSV manually using Python, we can just take advantage of pandas! Let's go ahead and import it!

# In[6]:

import pandas as pd


# We'll use **read_csv** and make note of the **sep** argument, we can also specify the desired column names by passing in a list of *names*.

# In[7]:

messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# ## Exploratory Data Analysis
# 
# Let's check out some of the stats with some plots and the built-in methods in pandas!

# In[8]:

messages.describe()


# Let's use **groupby** to use describe by label, this way we can begin to think about the features that separate ham and spam!

# In[9]:

messages.groupby('label').describe()


# 
# Let's make a new column to detect how long the text messages are:

# In[10]:

messages['length'] = messages['message'].apply(len)
messages.head()


# ### Data Visualization
# Let's visualize this! Let's do the imports:

# In[11]:

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')


# In[12]:

messages['length'].plot(bins=50, kind='hist') 


# In[13]:

messages.length.describe()


# Woah! 910 characters, let's use masking to find this message:

# In[14]:

messages[messages['length'] == 910]['message'].iloc[0]


# Looks like we have some sort of Romeo sending texts! But let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:

# In[18]:

messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# Very interesting! Through just basic EDA we've been able to discover a trend that spam messages tend to have more characters. (Sorry Romeo!)
# 
# Now let's begin to process the data so we can eventually use it with SciKit Learn!

# In[19]:

import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)


# Now let's see how to remove stopwords. We can impot a list of english stopwords from NLTK

# In[20]:

from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[21]:

nopunc.split()


# In[22]:

# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[23]:

clean_mess


# Now let's put both of these together in a function to apply it to our DataFrame later on:

# In[24]:

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# Here is the original DataFrame again:

# In[25]:

messages.head()


# Now let's "tokenize" these messages. Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).
# 
# Let's see an example output on on column:
# 
# 

# In[26]:

# Check to make sure its working
messages['message'].head(5).apply(text_process)


# In[27]:

# Show original dataframe
messages.head()


# ## Vectorization

# In[28]:

from sklearn.feature_extraction.text import CountVectorizer


# There are a lot of arguments and parameters that can be passed to the CountVectorizer. In this case we will just specify the **analyzer** to be our own previously defined function:

# In[31]:

# Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])

# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new `bow_transformer`:

# In[32]:

message4 = messages['message'][3]
print(message4)


# Now let's see its vector representation:

# In[34]:

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# This means that there are seven unique words in message number 4 (after removing common stop words). Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:

# In[36]:

print(bow_transformer.get_feature_names()[4073])
print(bow_transformer.get_feature_names()[9570])


# Now we can use **.transform** on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:

# In[39]:

messages_bow = bow_transformer.transform(messages['message'])


# In[40]:

print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[46]:

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[48]:

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# We'll go ahead and check what is the IDF (inverse document frequency) of the word `"u"` and of word `"university"`?

# In[50]:

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# To transform the entire bag-of-words corpus into TF-IDF corpus at once:

# In[51]:

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# ## Training a model

# We'll be using scikit-learn here, choosing the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier to start with:

# In[52]:

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# Let's try classifying our single random message and checking how we do:

# In[54]:

print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# Fantastic! We've developed a model that can attempt to predict spam vs ham classification!
# 
# ## Part 6: Model Evaluation
# Now we want to determine how well our model will do overall on the entire dataset. Let's begin by getting all the predictions:

# In[55]:

all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[56]:

from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))

