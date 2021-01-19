[Back to Main](/README.md/)

### Train Classifier to Predict the 'Sentiment' Class

#### Further Research on Topic

* Through researching dataset definitions in an above section, I was actually able to find an entire solution set to this assignment [at this blog here](https://harrisonjansma.com/apple), including tokenization, which already claimed an 89% accuracy using linear model using Logistic Regression, claiming a performance lead over a linear Support Vector Machine model.

[This article](https://machinelearningmastery.com/gentle-introduction-bag-words-model/) talks about the bag of words model in general.

> Word Hashing
You may remember from computer science that a hash function is a bit of math that maps data to a fixed size set of numbers. For example, we use them in hash tables when programming where perhaps names are converted to numbers for fast lookup. We can use a hash representation of known words in our vocabulary. This addresses the problem of having a very large vocabulary for a large text corpus because we can choose the size of the hash space, which is in turn the size of the vector representation of the document. Words are hashed deterministically to the same integer index in the target hash space. A binary score or count can then be used to score the word. This is called the “hash trick” or “feature hashing“. The challenge is to choose a hash space to accommodate the chosen vocabulary size to minimize the probability of collisions and trade-off sparsity.

[This article](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/) discusses the importance of defining a vocabulary for the bag-of-words model:

> It is important to define a vocabulary of known words when using a bag-of-words model. The more words, the larger the representation of documents, therefore it is important to constrain the words to only those believed to be predictive. This is difficult to know beforehand and often it is important to test different hypotheses about how to construct a useful vocabulary.



#### Pivoting to Linear Regression, Bag of Words Model

We used the above linear regression model to extract features from our previously generated, small golden dataset.  

##### Split into Random Subsets and Fit Vectorized Data According to TFIDF

```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


s1 = transformerready_df['nonstopwords_tokens_string']
s2 = sentimentmatrixgoldennonneutralrelevant_df['sentiment:confidence']

X = pd.concat([s1,s2], axis=1)
y = transformerready_df['sentiment']

#splitting data for cross validation of model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=False)

#Keeping the assignment confidence for later
X_train_conf, X_test_conf = X_train['sentiment:confidence'], X_test['sentiment:confidence']
X_train, X_test = X_train['nonstopwords_tokens_string'], X_test['nonstopwords_tokens_string']

#saving to csv
X_train.to_csv('train_clean.csv')
X_test.to_csv('test_clean.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

```

So what is going on in the above here?

```
from sklearn.model_selection import train_test_split
```
Here we are importing the module [model_selection](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) which splits arrays or matrices into random train and test subsets.

The way this is used in the code is:

```
train_test_split(X, y, test_size = 0.2,shuffle=False)
```

Basically, we pass 'X,' which is an array of all of our, "nonstopwords_tokens_string," which is text that has been preprocessed, next to a column of confidence values indicating a range of [0,1] describing the confidence of the assignment.

We also pass 'y,' which is the human-ascribed sentiment. These are both indexables with the same length.  The other parameters:

* test_size = 0.2 ... represents the proportion of the dataset to include in the test split.  So basically 20% of the data is put into training data, while the remaining 80% is left for testing data.
* shuffle = False ... basically, we keep the data ordinal, don't shuffle the data.

So after this, the data is split into test and training sets, X_train, X_test, y_train, y_test.

Then, the data is stored into appropriate variable names X_train_conf, X_test_conf for the confidences and X_train and X_test for the word tokens.

```
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
X_train_tfidf = vect.fit_transform(X_train) # transform categorical data, then fit to data provided
X_test_tfidf = vect.transform(X_test) # transform categorical data only (on test data)

```
Above, [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) takes the collection of raw documents, X_train and X_test and converts it into a matrix of TF-IDF features.  So basically, it is creating a vocabulary list for each set.  As we had talked about in [TFIDF](/assets/readmesections/TFIDF.md), this is basically a sparse matrix of class numpy.float64, stored in Compressed Sparse Row format.

Once again, TF is simply a calculation of the term frequency in a document, whereas IDF is the inverse of the document frequency which measures the informativeness of term t.  This is a measurement of signal-to-noise importance of a particular term across documents.

<hr>

vect.fit(data) means to fit the model to the data being provided. This is where the model "learns" from the data.

vect.transform(data) means to transform the data (produce model outputs) according to the fitted model.  Basically this means, categorical data is converted to a format a machine learning model can read.

vect.fit_transform(data) means to do both - Fit the model to the data, then transform the data according to the fitted model. Calling fit_transform is a convenience to avoid needing to call fit and transform sequentially on the same input.

<hr>

So what we end up with then is `X_train_tfidf` which is a fitted model according to X_train data, which was a random 20% of the original data.

##### Building a Linear Model and SVM Model, Comparing them and Exporting the Preferred

We then built a linear model with sklearn LogisticRegression, as shown below.

```
from sklearn.linear_model import LogisticRegressionCV

logr = LogisticRegressionCV()
logr.fit(X_train_tfidf, y_train)
y_pred_logr = logr.predict(X_test_tfidf)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

clf = SVC(class_weight = 'balanced')
pipe = Pipeline([('classifier', clf)])
fit_params = {'classifier__kernel':['rbf', 'linear', 'poly'],
          'classifier__degree':[2, 3, 4],
          'classifier__C':[0.01, 0.1, 1, 10]}

gs = GridSearchCV(pipe, fit_params, cv = 10, return_train_score = True)
gs.fit(X_train_tfidf, y_train)

print('Best performing classifier parameters (score {}):\n{}'.format(gs.best_score_, gs.best_params_))

pipe.set_params(classifier__degree = gs.best_params_['classifier__degree'],
                classifier__kernel = gs.best_params_['classifier__kernel'],
               classifier__C = gs.best_params_['classifier__C'])
pipe.fit(X_train_tfidf, y_train)
y_pred = pipe.predict(X_test_tfidf)
```

From this model, we can take a look at what the top negative sentiment and positive sentiment words were, in histogram format:

![Most Important Words Analysis](/assets/images/positivenegativecounts.png)

###### Importance of Stop Words

Note in the above that both the, "Negative Sentiment" words and "Positive Sentiment," words include a number of stop words, with Negative being, "you, this, me, my, on, and, why, an" - with only, "apple, fucking" being clearly unique terms. Likewise, "Positive Sentiment" words include, "all, for" with unique words clearly indidcating positive sentiment including, " king, love, top, tablets" and more questionable words including, "cnbctv, appl, applewatch."

Also notable is that the term, "apple" was considered negative, while, "appl," the stock term, was considered positive. Presumably, there might be some sort of stock pumping going on. This is an example of where subject matter expertise around the topic at hand can be important to understand, "true sentiment," vs. "self-optimized sentiment."

We can also plot a 2D visual of word frequencies, categorized by positive and negative. Note that our dataset is fairly sparse compared to existing blog posts utilizing this dataset.

![Word Frequencies](/assets/images/wordfrequencies.png)

###### Removing Stopwords

Since we have found from the above that our model will likely, intuitively not be of very high quality due to the high number of stop words used, it's necessary to come up with a way of removing stopwords. Of course we may not have found that we have a lot of stop words until this point in the analysis, after our, "most popular words," were already identified, but it will be necessary to go back to the original, preprocessed tweets and do additional processing to ensure that

```
pos_tweets = [('I love this car', 'positive'),
    ('This view is amazing', 'positive'),
    ('I feel great this morning', 'positive'),
    ('I am so excited about the concert', 'positive'),
    ('He is my best friend', 'positive')]

test = pd.DataFrame(pos_tweets)
test.columns = ["tweet","class"]
test["tweet"] = test["tweet"].str.lower().str.split()

from nltk.corpus import stopwords
stop = stopwords.words('english')
```
Suggests using list comprehension as a method to remove stopwords in a Pandas dataframe.  At the point of writing this analysis, it is reasonable to think that we could write code to perform the same list comprehension on our tokenized list as well.

```
test['tweet'].apply(lambda x: [item for item in x if item not in stop])

# yields:

0               [love, car]
1           [view, amazing]
2    [feel, great, morning]
3        [excited, concert]
4            [best, friend]
```

##### Our Code to Remove Stopwords:

We went back into our original model training Colab notebook and eliminated stopwords like so:

```
#Importing StopWords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

transformerready_df['nonstopwords_text'] = transformerready_df['text'].apply(lambda x: [item for item in x.split() if item not in stop])

```

![Non Stop Words Column Added](/assets/images/nonstopwords.png)

After removing stopwords, and also forcing lowercase for all words, we were able to create a vector which had seemingly better results, based upon a review of the top 10 most important words. The graph of word frequencies  also looked a bit more sensible:

![Improved Top Words](/assets/images/improvedtopwords.png)
![Improved Word Frequencies](/assets/images/improvedwordfrequencies.png)

##### Analyzing our Model Building Code

[LogisticRegressionCV()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html) is Logistic Regression with Cross-Validation classifier.  Logistic Regression is a popular statistical model. Of course, statistical models attempt to understand the relationship between input and output variables by estimating a probabilistic model from observed data.

All statistical methods follow the general formula:

```
y = f(x)+error
```
Where f(x) is the model, and error is the zero-mean Gaussian error.  Linear regression would be a form of statistical modeling in which the regression model is mx+b. The objective is to find parameters m and b that minimizes the error, or basically minimizing the mean square error.

Logistic regression assumes that there are two classes, either 0 or 1, and that samples from each class will fall within a normal, gaussian distribution centered around either 0 or 1 at the mean.  Converting these into a ratio and then a sigmoid function, posterior probability can be modeled.  In other words:

```
ln(px/(1-py)) = mx+b

or, signmoid function

s(t) = 1/(1+exp(-t)) where t=mx+b
```
So basically, logistic regression is similar to linear regression, but it instead measures the impact of input variables on an exponential/logarithmic function of posterior probability.

There is a difference in assumptions between logistic regression and linear regression:

1. Logistic Regression assumes binary or binomial distribution, where y-values are generated by independent binary trials, e.g. every, "word" is its own test, independent of other words, with each outcome having its own probability, whereas Linear Regression assumes Gaussian Noise (completely random words) with zero mean and constant variance across the entire text.
2.  The posterior probability of Logistic Regression is measured proportionally-logarithmicly as shown above, whereas for Linear Regression it is formed as a
3.  Both data are statistically independent from one another for both models, but Logistic Regression is binary.

So that being said:

```
logr = LogisticRegressionCV()
logr.fit(X_train_tfidf, y_train)
```
The above does a [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html) fit to the X_train_tfidf (tfidf vocab) and y_train (sentiment).  So basically, we are building a model, based upon probability as discussed above, that explains the relationship between the sparse matrices, X_train_tfidf (vocab) and ytrain (sentiment).  So if you think of these variables as the sparce matrixes that they are, you can conceptualize that a probability model, kind of like, "valence clouds" could be built to connect the two matricies.

```
y_pred_logr = logr.predict(X_test_tfidf)
```
Then, the above here simply predicts class labels for samples in X.  So whereas we inserted our training data above to fit, now we predict.

The remainder of the CoLab notebook evaluates the performance of this model, and compares it to SVM.  Interestingly, the two models performed identically, which shows that there is not enough data.

```
from sklearn.metrics import accuracy_score, classification_report

#Logistic Regression Eval
print('Logistic Regression Accuracy: ', accuracy_score(y_test, y_pred_logr))
print('\nLogistic Classification Report: \n' , classification_report(y_test,  y_pred_logr))
```

The [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) module of sklearn.metrics computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  In this situaton, y_test are the ground truth labels, while y_pre_logr are the tested labels.

The [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) module of sklearn.metrics helps us build a report showing the main classification metrics.

Interpreting our report:

```
Logistic Regression Accuracy:  0.6666666666666666

Logistic Classification Report:
               precision    recall  f1-score   support

           0       0.67      1.00      0.80         8
           1       0.00      0.00      0.00         4

    accuracy                           0.67        12
   macro avg       0.33      0.50      0.40        12
weighted avg       0.44      0.67      0.53        12

```

Basically, for 0 we had a few true positives out of 8, and for 1 we had no true positives out of 4.

* Accuracy is simply: accuracy = correct_predictions / total_predictions
* recall = (true positive)/(true positive + false negative)
* precision = (true positive)/(true positive + false positive)
* f1-score = 2 * (precision * recall)/(precision + recall)
* support is the number of samples of the true response that lie in that class
* macro avg is a non-weighted average of the given column above, not taking into account support
* weighted avg is a weighted average of the given column, taking into account portion of support


### Creating a Vocabulary to Pass to Scoring for Application

At this point, it is important to note that we need to pass not only the machine learning model over from our training model, but also a vocabulary.

This vocabulary was generated as, "index_to_word" within our "get_most_important_features" function.  We can export this vocabulary as a CSV file, which will be downloadable within our application.

More about exporting the vocabulary is covered under, [Scoring](/readmesections/Scoring.md)




[Back to Main](/README.md/)
