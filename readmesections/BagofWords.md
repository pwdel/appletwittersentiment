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

#saving to csv
X_train.to_csv('train_clean.csv')
X_test.to_csv('test_clean.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

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

We can also plot a 2D visual of word frequencies, categorized by positive and negative. Note that our dataset is fairly sparse compared to existing blog posts utilizing this dataset.

![Word Frequencies](/assets/images/wordfrequencies.png)

[Back to Main](/README.md/)
