[Back to Main](/README.md/)

### Save the Model To Some Format For Scoring

Scoring is the process of applying an algorithmic model built from a historical dataset to a new dataset in order to uncover practical insights that will help solve a business problem, basically, "saving the model." The word, "scoring" is used in the sense of scoring a piece of metal to mark a future cut or drill, basically creating a future hopefully repetitive model.

The code snippet showing how this scoring was done using a logistic regression model is shown below:

```
# We had developed the above model fit as follows:
# logr = LogisticRegressionCV()
# logr.fit(X_train_tfidf, y_train)

# import joblib, used for scoring
from sklearn.externals import joblib
# use joblib.dump to dump the model into a .pkl file type
joblib.dump(logr, 'model.pkl')
```

After running this, the Logistic Regression model built in previous steps within this assignment is now persisted. This model can be loaded into memory with a single line of code. Loading the model back into a workspace is known as Deserialization.

The model.pkl can be saved to Google Drive as follows:

```
# Save this file to Google Drive

folder_id = "1WicGkBotOouPvv4pwAk1Frfj7xFOwKG4"

# get the folder id where you want to save your file
file = drive.CreateFile({'parents':[{u'id': folder_id}]})
file.SetContentFile('model.pkl')
file.Upload()
```

![Model Saving Location on Google Drive](/assets/images/savingmodel.png)

We can't stop completely at saving the model to Google Drive, we also have to save columns, since this is a sparse matrix, as well as the vocabulary.  Saving the vocabulary is done within the training Google Colab document via:

```
index_to_word_df = pd.DataFrame.from_dict(index_to_word,orient='index')
print(pd.DataFrame.from_dict(index_to_word,orient='index'))

import csv

from google.colab import drive
drive.mount('drive')

index_to_word_df.to_csv('index_to_word.csv')
!cp index_to_word.csv "drive/My Drive/"

```

With index_to_word_df representing the dataframe including our vocabulary.  The output CSV file for this looks like the following:

![Sample Vocab Index](/assets/images/samplevocabindex.png)

Looking through this overall list, we see that we have 296 values, which is massively scaled down from what would likely have been found without eliminating stop words and extra characters with regex. Of course, there are also opportunities for future improvement that can even be seen at this stage - including the elimination of numerals, which seem not to add much value, and the expansion of stop items to include additional online slang vocabulary, such as, "ur."

### How is the Scoring Model Actually Mathematically Used?

So while we know how to load a scored model into a filebase, the question is - once this particular file is loaded, how do you use it?  What does the model indicate and how can you leverage it?

#### Load Machine Learning Model into Memory

After loading the model.pkl file into memory, I printed it out to see what it might look like, and it showed the following:

```
LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=None,
                     penalty='l2', random_state=None, refit=True, scoring=None,
                     solver='lbfgs', tol=0.0001, verbose=0)
```

To get a better understanding of what this means, and to be able to work with it more thoroughly I [looked here at LogisticRegressionCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html). The, "CV," here stands for, "Cross Validation."

Logistic regression is a classification approach for different classes of data in order to predict whether a data point belongs to one class or another. Sigmoid hypothesis function is used to calculate the probability of y belonging to a particular class. Training data is normalized using Zscore.

![Logistic Regression Example](/assets/images/logistiregression.png)

To get a better idea of how a saved, .pkl model is used within an app, I reviewed the following article:

[Develop an NLP Model in Python and deploy with Flask](https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776)

From reading through the code and the overviews given in this article, there appears to be two main approaches:

> Inside [a] predict function, access [the data set], pre-process the text, and make predictions, then store the model. We access the new message entered by the user and use our model to make a prediction for its label.

The code is given as shown below, in which essentially a training model is set up dynamically each time the overall app.py function is called.

```
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
```

An alternate method is given in the code, commented out is shown below. So the, "alternative method essentially loads the spam model as a saved file into the variable, "clf," or classifier, just as it is with the method mentioned above.

```
#Alternative Usage of Saved Model
# joblib.dump(clf, 'NB_spam_model.pkl')
# NB_spam_model = open('NB_spam_model.pkl','rb')
# clf = joblib.load(NB_spam_model)
```
Both methods are followed by:
```
if request.method == 'POST':
  message = request.form['message']
  data = [message]
  vect = cv.transform(data).toarray()
  my_prediction = clf.predict(vect)
return render_template('result.html',prediction = my_prediction)
```
In either method:

1. The message from a user comes in, and is transformed into a vector with, "cv.transform" which is a method of [scikitLearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) shown here.

The documentation for scikit learn says:

> Convert a collection of text documents to a matrix of token counts  This implementation produces a sparse representation of the counts using scipy.sparse.csr_matrix.

> If you do not provide an a-priori dictionary and you do not use an analyzer that does some kind of feature selection then the number of features will be equal to the vocabulary size found by analyzing the data.

So basically, all cv.transform does is take the vocabulary at hand, given in message, it will just proceed with a vocabulary analysis on that message itself. So basically, cv.transform doesn't have, "awareness," of the previous model.

To use cv.transform properly and have the model make sense, the, "corpus" that gets entered into CountVectorizer() has to include both the "message" or, "user input" combined with a list of known words from the training analysis done earlier.  We can define that work as follows:

> [corpus] = [user input] + [training dictionary of all terms]

So given this, we have to ensure that we output our entire important vocabulary from the training method, and then upload it into our, "corpus" application during this stage.

Of course, we don't really know where these models are going to be deployed, and the computing power, but often the vocabulary should be, "limited," whatever that means, vs. the overall larger training set vocabulary, to reduce processing intensity.  The corpus vocabulary we had generated included 296 words, which seems sufficiently small.

Once we have a corpus built, which is essentially a list of strings, we can then put into the vectorizer via fit_transform(), as shown...

```
corpus = [
...     'This is the first document.',
...     'This document is the second document.',
...     'And this is the third one.',
...     'Is this the first document?',
... ]
>>> vectorizer = CountVectorizer()
>>> X = vectorizer.fit_transform(corpus)

```
From our training model discussion under the [BagofWords](/readmesections/BagofWords.md) section, we had originally exported a CSV with the entire vocabulary. However after attempting in different ways unsuccessfully to upload and download the CSV due to an encoding issue, we decided to simply create a list of all of the vocabulary words as, "vocab_list."


### Evaluating the Linear Regression Approach

[This article](https://machinelearningmastery.com/gentle-introduction-bag-words-model/) talks about the bag of words model in general.

> Word Hashing
You may remember from computer science that a hash function is a bit of math that maps data to a fixed size set of numbers. For example, we use them in hash tables when programming where perhaps names are converted to numbers for fast lookup. We can use a hash representation of known words in our vocabulary. This addresses the problem of having a very large vocabulary for a large text corpus because we can choose the size of the hash space, which is in turn the size of the vector representation of the document. Words are hashed deterministically to the same integer index in the target hash space. A binary score or count can then be used to score the word. This is called the “hash trick” or “feature hashing“. The challenge is to choose a hash space to accommodate the chosen vocabulary size to minimize the probability of collisions and trade-off sparsity.

https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/

> It is important to define a vocabulary of known words when using a bag-of-words model. The more words, the larger the representation of documents, therefore it is important to constrain the words to only those believed to be predictive. This is difficult to know beforehand and often it is important to test different hypotheses about how to construct a useful vocabulary. We have already seen how we can remove punctuation and numbers from the vocabulary in the previous section. We can repeat this for all documents and build a set of all known words. We can develop a vocabulary as a Counter, which is a dictionary mapping of words and their count that allows us to easily update and query.

Basically,

### Exporting a Vocab List



[Back to Main](/README.md/)
