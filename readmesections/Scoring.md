[Back to Main](/README.md/)

### Save the Model To Some Format For Scoring

Scoring is the process of applying an algorithmic model built from a historical dataset to a new dataset in order to uncover practical insights that will help solve a business problem, basically, "saving the model." The word, "scoring" is used in the sense of scoring a piece of metal to mark a future cut or drill, basically creating a future hopefully repetitive model.

```
# We had developed the above model fit as follows:
# logr = LogisticRegressionCV()
# logr.fit(X_train_tfidf, y_train)

# import joblib, used for scoring
from sklearn.externals import joblib
# use joblib.dump to dump the model into a .pkl file type
joblib.dump(logr, 'model.pkl')
```

The Logistic Regression model is now persisted. You can load this model into memory with a single line of code. Loading the model back into your workspace is known as Deserialization.

We save our model.pkl to Google Drive as follows:

```
# Save this file to Google Drive

folder_id = "1WicGkBotOouPvv4pwAk1Frfj7xFOwKG4"

# get the folder id where you want to save your file
file = drive.CreateFile({'parents':[{u'id': folder_id}]})
file.SetContentFile('model.pkl')
file.Upload()
```

![Model Saving Location on Google Drive](/assets/images/savingmodel.png)

[Back to Main](/README.md/)
