[Back to Main](/README.md/)

### Stand Up API Service That Has Inputs and Outputs

I found this article on [Turning Machine Learning Models into APIs in Python](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python), to get a general idea about how to approach this.  According to this tutorial, I should use Flask to stand up an API. [I have used Flask before](https://www.patdel.com/home-data-flask-app/) to build a web application and know that it is time-consuming.

However, pieces of this article were useful to build the API, including how scoring was accomplished above.

#### Pseudocode for API Service

0. Load Machine Learning Model into Memory
1. Ask for input text, 280 characters or less, restrict input.
2. Vectorize function which takes that input text and vectorizes it.
3. Prediction function which takes the vectorized object and applies the machine learning model to it, outputting a prediction and probability.
4. API function which turns the prediction result and probability result into a json object.
5. Print or output Json object to terminal including result.

#### Notes on Above Process


#### Input: Plain Text

Basically we set an input as follows.  We ask for an input, constraiing the input to lower case and 280 Characters.

```
# clear the tweet string, default to "a"
tweet_str = "a"
# ask for input
tweet_str = input("Enter Your Tweet (Maximum 280 Characters, lowercase letters a-z) : ")[:280] # truncate to 280 characters
# regex which includes space
if not re.match("^[a-z \S]*$", tweet_str):
  print("Error! Only letters A-z allowed!")
  sys.exit()

print("Your Tweet is: ",tweet_str)

# put tweet_str into dataframe

# Construct input to dataframe
data = {'tweetstring': [tweet_str]}
# put tweetstring into dataframe for use in tokenizer
tweetstr_df = pd.DataFrame(data, columns = ['tweetstring'])
```

#### Output: Prediction of Sentiment & Probability of Prediction

Import the model:

```
# load linear regression model back into memory
lr_model = joblib.load('model.pkl')

print(lr_model)

LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
                     fit_intercept=True, intercept_scaling=1.0, l1_ratios=None,
                     max_iter=100, multi_class='auto', n_jobs=None,
                     penalty='l2', random_state=None, refit=True, scoring=None,
                     solver='lbfgs', tol=0.0001, verbose=0)
```

Add tokens using bag of words method and regextokenizer.

```
# Add Tokens Using Bag of Words Method
from nltk.tokenize import RegexpTokenizer

#NLTK tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# apply tokenizer
the_tokens = tweetstr_df['tweetstring'].apply(tokenizer.tokenize)

# create the column to fill
tokens_df = pd.DataFrame({'tokens':the_tokens})
```

Adding dummies to tokens to make input viable.

```
# tokens_df is our dataframe containing the tokenized new tweet input
query_df = pd.get_dummies(tokens_df['tokens'][0])

# Conform Series/DataFrame to new index with optional filling logic.
query_df = query_df.reindex(columns=model_columns, fill_value=0)
```

Using lr_model to predict based upon our query which is the reindexed version shown above.

```
# print(regr.coef_)
# put together prediction of each word in a list
prediction = list(lr_model.predict(query_df))

# put together raw prediction in numbered array
prediction_raw = lr_model.predict(query_df)

# sum up the prediction
prediction_sum = np.sum(prediction_raw)
```

We have to put some logic in place which notates negative vs. positive sentiment based upon the predictor.

Output the prediction as an API / Json readable format.

```
import json

{
  tweet:
  prediction:
}
```

### Model Deployment

Docker is not available for the laptop that I brought with me on my trip.  I brought a Macbook Air running macOS High Sierra, and evidently Docker Requires either Mojave or something better than a Macbook Air.

So, rather than create a docker model, I just created a virtual environment using virtualenv.  This is somewhat similar to Docker, in that it's a way of creating a virtual environment, with instructions on what modules to install in a requirements.txt document.  The disadvantage is of course it is not as deployable, repeatable, and has the potential for creating bloat, as opposed to Docker based models which allow a developer to go more, "thin" in terms of the languages, packages and dependencies, as well as having the code itself dockerized.

I understand that containers are critical for modern enterprise app development, but for the purposes of getting this code done and shipped, I had to pivot given the machine I have.  This can always be dockerized in the future.

The way to, "save" dependencies as one goes along using a virtual environment within python is to continuously run,

```
pip3 freeze > requirements.txt
```

I wrote this application using super sloppy coding, with no functions, just wanted to get a demo app going.

### Toolchain Debugging

When attempting to build our app from the CoLab source, we run into an error relating to Scipy. There appears to be a [toolkit variability between scipy and numpy as documented here](https://docs.scipy.org/doc/scipy/reference/toolchain.html).

> The table shows the NumPy versions suitable for each major Python version (for SciPy 1.3.x unless otherwise stated).

```
$ pip3 uninstall scipy
$ pip3 install scipy==1.3
```

### Additional Code

Find [here]()

### Deployment Instructions

Can be found at [README.md](/README.md/)

[Back to Main](/README.md/)
