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

## My Plan of Attack

The first general rule for building anything at all is to understand its utility. In this case, I wanted to first understand as much as I could, within reason, about the, "why," behind this project. My career history includes a decent amount of sales, business development and tech community organizing, so typically my first approach to any project is to try to see what the, "why behind the why," might be for a particular customer or stakeholder request.  So, I asked to see if there were any additional considerations:

![01](/assets/images/01.png)

Of course, I didn't want to get too deep into the weeds, since this is after all - a part of a job interview process. The best approach is most likely to demonstrate directly what is being asked, which was as follows:

> The final result should be a Python script that we can run locally to replicate your work. We would like to be able to understand your approach to training the model as well as the steps you take to achieve the best accuracy of the model.  Once you complete the assignment, please send it back to _NAME_ (_NAME_@company.com) with any instructions on how to run the script. If you advance to the on-site interview, we’ll ask you to give a short presentation about your approach and other techniques you would have tried if you had more time.

Further to this, it was specified that:

> Even though at _Company_ we dedicate a lot of effort to making our ML models as accurate as possible, for this assignment we are most interested in the process of how you piece all of these components together.

So given all of the above, and my background, I decided upon the following approach to building something out:

1. Use [Google CoLab](https://colab.research.google.com/) to quickly attempt to build something that barely works within as short of a time as possible.  
** I tend to angle toward prototyping in notebooks or IDEs if one exists for a particular problem set rather than in straight code. I believe this makes the process of thinking and iterating more efficient, by abstracting away library management and anything related to devops as much as possible early in the process, but rather to focus on driving value immediately. Part of the reason why I would select Colab rather than using a Jupyter notebook is because I don't have to worry about setting up a Python environment, or worrying about which Python environment needs to fit later in the Machine Learning pipeline at this stage, it's just straight code.
2. Do background research and decide upon naming conventions, do light research on approach as I dive into the analysis. Sketch out thoughts, any math needed, poems jokes, songs or comics that come to mind which are relavant to the work at hand.
3. Examine the output and overall codebase. Review and think about what needs to be done next to meet all of the criteria above.
4. Iterate and build and deploy in a dockerized format, per the assignment request.
5. Refine the above according to anything thought provoking that I may think of or come across along the way.
6. Write up the report within the README file.

## Building the Prototype

* In this step, I opted to use Google CoLab.  There is one perhaps slightly annoying devops-related setup step in using CoLab, in that you have to authenticate a Google API Python Client to be able to perform basic Google Drive API tasks within CoLab, including accessing our CSV file.

![installing pydrive](/assets/images/installing-pydrive.png)

## Requested Steps from Work Assignment
### Reading the Dataset into Memory

* In this step, my in

### Compute Set of Engineered Features
### Use regex functions to extract some specific terms
### Compute word embeddings (i.e. word2vec, GloVe, BERT)
### Tfidf
### Train Classifier to Predict the 'Sentiment' Class
### Save the Model To Some Format For Scoring
### Stand Up API Service That Has Inputs and Outputs
#### Input: Plain Text
#### Output: Prediction of Sentiment & Probability of Prediction
### Model Deployment

Dockerizing Colab

SSH into Colab


3.	Trains any classifier to predict the `sentiment` class
4.	Saves the model to some format that can be used for scoring
5.	Stand-up an API service that has:
a.	Input: plain text
b.	Output: prediction of sentiment + the probability of that prediction
6.	Deployment of model + api service should be dockerized

Docker
