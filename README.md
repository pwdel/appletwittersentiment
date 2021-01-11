# Apple Twitter Sentiment Analysis

Apple Twitter Sentiment Analysis Project

## Meta Objective

* The objective of this report is to clearly and concisely document my thoughts working on, "The Apple Computers Twitter Sentiment dataset," project.
* This work is being documented under [Tom Preston Warner's README driven development](https://tom.preston-werner.com/2010/08/23/readme-driven-development.html) methodology.

## Project Objective

* Create a machine learning pipeline, train a model and optionally stand-it up for inference.
* Demonstrate your ability to create a machine learning pipeline, train a model and optionally stand-it up for inference.
* Allow Carvana to understand the process of how you piece all of these components together.

## My Background

* I have not ever performed any kind of sentiment analysis previously.  I am vaguely familiar with the mathematics from having watched conference talks, but in essence have no idea what I'm doing.
* My main languages in order of skill are Matlab, Python and Ruby.  My initial thought is that Python is the way to go here, though the work statement mentions, "use any language you are comfortable with."  This is based upon apriori knowledge of the team's tech stack using Python.  
** Ruby: I could do a short analysis of how feasible this would be, and how one would go about building this in Ruby.
** Matlab: From a quick review, Matlab appears to have the greatest amount of documentation and prebuilt, ready-to-go framework around many facets of textual analysis including sentiment analysis, as it does with most things, but Matlab is not a deployable platform per se, not open source, expensive, and plugins or "toolboxes" as they call them, are an additional expense.  Matlab can be considered more of a research tool.
* I am not familiar with any of the standard plugins used in sentiment analysis via Python.

## Suggested Steps from Work Assignment
### Reading the Dataset into Memory
### Compute Set of Engineered Features
### Use regex functions to extract some specific terms
### Compute word embeddings (i.e. word2vec, GloVe, BERT)
### Tfidf
### Train Classifier to Predict the 'Sentiment' Class
### Save the Model To Some Format For scoring
### Stand Up API Service That Has Inputs and Outputs
#### Input: Plain Text
#### Output: Prediction of Sentiment & Probability of Prediction
### Model Deployment


3.	Trains any classifier to predict the `sentiment` class
4.	Saves the model to some format that can be used for scoring
5.	Stand-up an API service that has:
a.	Input: plain text
b.	Output: prediction of sentiment + the probability of that prediction
6.	Deployment of model + api service should be dockerized
