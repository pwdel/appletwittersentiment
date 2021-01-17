[Back to Main](/README.md/)

### Save the Model To Some Format For Scoring

Scoring is the process of applying an algorithmic model built from a historical dataset to a new dataset in order to uncover practical insights that will help solve a business problem, basically, "saving the model." The word, "scoring" is used in the sense of scoring a piece of metal to mark a future cut or drill, basically creating a future hopefully repetitive model.

The code snippet showing how this scoring was done is shown below:

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

To get a better understanding of what this means, and to be able to work with it more thouroughly I [looked here at LogisticRegressionCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html). The, "CV," here stands for, "Cross Validation."

Logistic regression is a classification approach for different classes of data in order to predict whether a data point belongs to one class or another. Sigmoid hypothesis function is used to calculate the probability of y belonging to a particular class. Training data is normalized using Zscore.

![Logistic Regression Example](/assets/images/logistiregression.png)

To get a better idea of the definitions contained:

* Cs - Each of the values in Cs describes the inverse of regularization strength. If Cs is as an int, then a grid of Cs values are chosen in a logarithmic scale between 1e-4 and 1e4. Like in support vector machines, smaller values specify stronger regularization.
* class_weight
* cv
* dual
* fit_intercept - Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
* intercept_scaling
* l1_ratios
* max_iter
* multi_class
* n_jobs
* penalty
* random_state
* refit
* scoring
* solver
* tol
* verbose


![Model Saving Location on Google Drive](/assets/images/savingmodel.png)

[Back to Main](/README.md/)
