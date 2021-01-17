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

##### Load Machine Learning Model into Memory

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

#### Input: Plain Text



#### Output: Prediction of Sentiment & Probability of Prediction

* get_dummies - Turn a categorical variable into a series of zeros and ones, which makes them a lot easier to quantify and compare.
* pandas.DataFrame.reindex - Conform Series/DataFrame to new index with optional filling logic

### Model Deployment

[Back to Main](/README.md/)
