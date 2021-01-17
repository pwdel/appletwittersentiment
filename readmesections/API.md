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



#### Output: Prediction of Sentiment & Probability of Prediction

* get_dummies - Turn a categorical variable into a series of zeros and ones, which makes them a lot easier to quantify and compare.
* pandas.DataFrame.reindex - Conform Series/DataFrame to new index with optional filling logic

### Model Deployment

[Back to Main](/README.md/)
