import re
import pandas as pd
import pickle
import joblib

# Add Tokens Using Bag of Words Method
from nltk.tokenize import RegexpTokenizer

# sklearn
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------- Install Models


with open('model_columns.pkl', 'rb') as f:
    model_columns = joblib.load(f)

with open('model.pkl', 'rb') as f:
    model = joblib.load(f)


# ------------------------------- Input

print("Welcom to the Draft Apple Tweet Sentiment Analysis App")

# clear the tweet string, default to "a"
tweet_str = "a"
# ask for input
tweet_str = input("Enter Your Tweet (Maximum 280 Characters, lowercase letters a-z) : ")
# regex which includes space
if not re.match("^[a-z \S]*$", tweet_str):
  print("Error! Only letters A-z allowed!  Please fix and run program again.")
  sys.exit()

print("Your Tweet string is: ",tweet_str)

# Construct input to dataframe
data = {'tweetstring': [tweet_str]}
# put tweetstring into dataframe for use in tokenizer
tweetstr_df = pd.DataFrame(data, columns = ['tweetstring'])

print("The dataframe format of the tweetstring is: ",tweetstr_df)

# ------------------------------- Tokenizer

#NLTK tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# apply tokenizer
the_tokens = tweetstr_df['tweetstring'].apply(tokenizer.tokenize)

# create the column to fill
tokens_df = pd.DataFrame({'tokens':the_tokens})
print(tokens_df)

# ------------------------------- Run Prediction

# tokens_df is our above dataframe
query_df = pd.get_dummies(tokens_df['tokens'][0])

print(model_columns)

# Conform Series/DataFrame to new index with optional filling logic.
query_df = query_df.reindex(columns=model_columns, fill_value=0)
# print(regr.coef_)
prediction = list(lr_model.predict(query_df))

prediction_raw = lr_model.predict(query_df)

prediction_sum = np.sum(prediction_raw)

print("prediction is:",prediction)
print("prediction length is:",len(prediction))
print("raw prediction is: ",(prediction_raw))
print("prediction sum is: ",(prediction_sum))

# ------------------------------- Run Report

# report numbers will be used to determine false negatives and positives for probability metric

# ------------------------------- Print API Json Output

if prediction_sum>0
    sentiment_json = { "originaltweet": {},"sentiment": {"positive"},"probability": {}}
    print(sentiment_json)
else
    sentiment_json = { "originaltweet": {},"sentiment": {"negative"},"probability": {}}
    print(sentiment_json)
