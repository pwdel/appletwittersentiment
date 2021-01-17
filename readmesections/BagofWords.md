[Back to Main](/README.md/)

### Train Classifier to Predict the 'Sentiment' Class

#### Further Research on Topic

* Through researching dataset definitions in an above section, I was actually able to find an entire solution set to this assignment [at this blog here](https://harrisonjansma.com/apple), including tokenization, which already claimed an 89% accuracy using linear model using Logistic Regression, claiming a performance lead over a linear Support Vector Machine model.

#### Pivoting to Linear Regression, Bag of Words Model

We used the above linear regression model to extract features from our previously generated, small golden dataset.  

```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

s1 = transformerready_df['text']
s2 = sentimentmatrixgoldennonneutralrelevant_df['sentiment:confidence']

X = pd.concat([s1,s2], axis=1)
y = transformerready_df['sentiment']

#splitting data for cross validation of model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=False)

#Keeping the assignment confidence for later
X_train_conf, X_test_conf = X_train['sentiment:confidence'], X_test['sentiment:confidence']
X_train, X_test = X_train['text'], X_test['text']

print(X_train[:5])

```

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

Note in the above that both the, "Negative Sentiment" words and "Positive Sentiment," words include a number of stop words, with Negative being, "you, this, me, my, on, and, why, an" - with only, "apple, fucking" being clearly unique terms. Likewise, "Positive Sentiment" words include, "all, for" with unique words clearly indidcating positive sentiment including, " king, love, top, tablets" and more questionable words including, "cnbctv, appl, applewatch."

Also notable is that the term, "apple" was considered negative, while, "appl," the stock term, was considered positive. Presumably, there might be some sort of stock pumping going on. This is an example of where subject matter expertise around the topic at hand can be important to understand, "true sentiment," vs. "self-optimized sentiment."

We can also plot a 2D visual of word frequencies, categorized by positive and negative. Note that our dataset is fairly sparse compared to existing blog posts utilizing this dataset.

![Word Frequencies](/assets/images/wordfrequencies.png)

#### Removing Stopwords

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

### Our Code to Remove Stopwords:
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




[Back to Main](/README.md/)
