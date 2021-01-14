# Apple Twitter Sentiment Analysis

Apple Twitter Sentiment Analysis Project

## Meta Objective

* The objective of this report is to clearly and concisely document my thoughts working on, "The Apple Computers Twitter Sentiment dataset," [project assignment](/assets/assignment/assignment.md).
* This work is being documented under [Tom Preston Warner's README driven development](https://tom.preston-werner.com/2010/08/23/readme-driven-development.html) methodology.

## Project Objective

* Create a machine learning pipeline, train a model and optionally stand-it up for inference.
* Demonstrate your ability to create a machine learning pipeline, train a model and optionally stand-it up for inference.
* Allow _Company_ to understand the process of how I piece all of these components together and what steps I take to ensure accuracy.

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
2. Do background research, analysis and deeper dive into the data itself to understand to the greatest degree what is contained in the dataset, and decide upon naming conventions. This also involves doing light research on an approach to build an app as I dive into the analysis. This may also involve some back and fourth questions with the team providing the assignment, which may change depending upon the context of who the dataset was sent from, and what the project purpose is. There are always differences in context which dictate how much communication is called for, vs. how much independent work is expected to maximize overall team efficiency.
3. Examine the output and overall codebase. Review and think about what needs to be done next to meet all of the criteria above.
4. Iterate and build and deploy in a dockerized format, per the assignment request.
5. Refine the above according to anything thought provoking that I may think of or come across along the way.
6. Write up the report within the README file.

## Building the Prototype

* In this step, I opted to use Google CoLab.  There is one perhaps slightly annoying devops-related setup step in using CoLab, in that you have to authenticate a Google API Python Client to be able to perform basic Google Drive API tasks within CoLab, including accessing our CSV file.

![installing pydrive](/assets/images/installing-pydrive.png)

* I'm opting out of test based development at this point. The objective is quality of analysis and speed of development time.
* In addition to PyDrive being installed, any files that we read from on Drive itself must be open and shared with the world, at least in non-GSuite instances of this methodology, which in a normal production environment pose a security risk, akin to the all-too common unsecured public S3 bucket.

## Requested Steps from Work Assignment
### Reading the Dataset into Memory

* Within the prototyping phase, the first step of reading the data into memory was to simply download the content as a string object.

```
downloaded.GetContentFile('Apple-Twitter-Sentiment-DFE.csv')  
```

* Putting the data into a more organized fashion involves Pandas. The .csv file did not appear to be latin-1 encoded.

```
sentiment_mix = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin-1')
```

#### Dialing into the Data Columns

At this point, since we are able to actually load the data into the system, before we actually start cleaning the data, it might be appropriate to understand a bit more about what the various columns mean. We don't have defined instructions.

I found a [description of the original dataset here](https://data.world/crowdflower/apple-twitter-sentiment), which states the following:

> A look into the sentiment around Apple, based on tweets containing #AAPL, @apple, etc. Contributors were given a tweet and asked whether the user was positive, negative, or neutral about Apple. (They were also allowed to mark "the tweet is not about the company Apple, Inc.)

In addition, this [Kaggle Competition](https://www.kaggle.com/c/apple-computers-twitter-sentiment2/data) mentions the following about the dataset:

> In .csv files they are idenitified as: 1) Positive - 5; 2) Neutral - 3; 3) Negative - 1.

#### Dialing Deeper Into What the Data Actually Means

##### My Understanding of the Setup

Based upon the above description that, "Contributors were given a tweet and asked whether the user was...," it appears that this spreadsheet was built in two to three "stages."

* The first stage appeared to involve "Contributors," ascribing a positivity, negativity or neutral rating the text data within the, 'sentiment' column.
* The second stage appeared to involve applying some sort of "judgement" layer to that original sentiment rating. It appears that there may have been, "Meta Contributors," ascribing some sort of manual judgement on top of the, "Contributors," as a form of dual-factor authentication, within the column titled, "_trusted_judgments".  However, it is also possible that somehow the Contributors themselves, rather than Meta Contributors. There appears to have been a, _last_judgement_at date recorded within the named column, which appears to refer to the, _trusted_judgement column, although it is entirely possible that the _last_judgement_at column is referring to the, "sentiment" judgement.  Based upon the locations of the date columns, we can reasonably believe there were two stages, but this would be a good question to normally ask to whomever provided the data originally.


| Column               | Interpretation                                                                                     |
|----------------------|----------------------------------------------------------------------------------------------------|
| _unit_id             | appears to be an arbitrarily assigned, ordinal listing of all rows.                                                                                                     |
| _golden              | TRUE can be assumed as being 100% accurate, while FALSE cannot.                                    |
| _unit_state          | "golden" can be assumed as being 100% accurate, while "finalized" can be assumed as being unknown. |
| _trusted_judgments   |                Appears to be a scaled measurement of some kind, ranging from a minimum of 3 to a maximum of 27.  This appears to be some kind of measurement that was put in place to score either how trusted an entire column sentiment judgement is from a meta user, or how much a particular user trusted their own judgment                                                                                     |
| _last_judgment_at    | Appears to be either when the last, _trusted_judgement was applied. If our assumption that _trusted_judgements were applied in a, "Second Stage" by a "Meta Contributor" this makes sense  because this date occurs after what appears to be the date that the sentiment was ascribed, if indicated by column "date" Much of the "_last_judgement_at" space is nullspace, the reasons for this are unknown.                                                                                                   |
| sentiment            | sentiment is the score assigned by Contributors, who assigned optionally 1, 3, or 5, which map out to -1, 0, and +1 in a standard normalized sentiment algorithm.                                                                                                   |
| sentiment:confidence | 654 possible values, evidently originally normalized, ranging from 0.3327 to 1. This evidently refers to the confidence                                                                                                     |
| date                 | Appears to indicate the date that the sentiment  was ascribed. This date appears to                                                                                                      |
| id                   | Appears to be some overarching id number, all rows contain the value '540000000000000000'                                                                                                   |
| query                |           Appears to be the query entered to come up with the text row result. All values were #AAPL OR @Apple                                                                                         |
| sentiment_gold       | Appears to be an unstructured column that contains sentiment values, but some of the sentiment values contain two sentiment measurements, as well as the term "not_relevant" One row contains what appears to have originally been raw text.  All possible values include: "1", "3", "5","3 1", "3 1 not_relevant", "3 not_relevant", "@tschwettman @Apple THIS IS THE WORST DAY OF MY LIFE","5 3 1","5 3 not_relevant"  It is unknown why this column is termed, "gold"                                                                                                   |
| text                 | The raw, original text strings                                                                     |

#### Further Filtering the Data

*What does the _golden column sample mean?*

Inaccurate records can be a significant burden on organizational productivity. The idea behind, 'The Golden Record,' according to data management principals, is to have a single source of truth, assumed to be 100% accurate by multiple parts of an organization. Since research teams do not work in a vacuum, and there are always monetary considerations for any ongoing work, it is important to at least ask other stakeholders the importance of a particular notation to understand how critical it may be for a particular analysis.  Typically, "golden record" indicates that accuracy for non-golden record classified data points is suspect, from the standpoint of technology management.

From this standpoint, it would appear that it may be important to eliminate all non-golden samples.  While a particular algorithm built may, "self report," its own accuracy based upon input data alone, there is another question as to whether this model performance actually reflects what happens in objective reality. In other words:

> What right do we have to say that our model actually describes the desires of our customers?

### Compute Set of Engineered Features


### Use regex functions to extract some specific terms

Identification of Regex Functions could be done on specific terms, but if we are looking at this entire project holistically, we can observe that the dataframe columns include all sorts of interesting info, including time series information which could be utilized in different interesting ways in the future. Since this project may possibly include requests for future research, it would be interesting to see how to perhaps, "bulk clean," the entire text column in a way that makes the data more accessible for future project iterations.

Intuitively, Twitter is a fairly well known web element, and likely there are some pre-existing libraries of regex's out there which we may be able to use. A cursory investigation yielded this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).

[Python-Twitter-API](https://python-twitter.readthedocs.io/en/latest/twitter.html)
[Tweepy](https://www.tweepy.org/)

https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression


https://ieeexplore.ieee.org/document/8022667

https://datascience.stackexchange.com/questions/30516/how-does-one-go-about-feature-extraction-for-training-labelled-tweets-for-sentim



### Compute word embeddings (i.e. word2vec, GloVe, BERT)


#### Researching Steps to Compute Word Embeddings

word2vec

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb

GloVe

https://colab.research.google.com/github/mdda/deep-learning-workshop/blob/master/notebooks/5-RNN/3-Text-Corpus-and-Embeddings.ipynb

BERT

https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb

### Tfidf

https://colab.research.google.com/drive/1h6Jpgcdv2kB07zkcLKFpFM9xsSiZE9pU

### Train Classifier to Predict the 'Sentiment' Class

#### Further Research on Topic

* Through researching dataset definitions in an above section, I was actually able to find an entire solution set to this assignment [at this blog here](https://harrisonjansma.com/apple), including tokenization, which already claimed an 89% accuracy using linear model using Logistic Regression, claiming a performance lead over a linear Support Vector Machine model.




Using features extracted above, we can assign positive and negative values to words.

Is there a ready made database of positive and negative words which may already exist within our text?

https://www.quora.com/Is-there-a-downloadable-database-of-positive-and-negative-words

Subjectivity Lexicon

http://mpqa.cs.pitt.edu/

Another sentiment Lexicon
https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon

https://github.com/apmoore1/SentiLexTutorial

https://github.com/apmoore1/SentiLexTutorial/blob/master/Tutorial.ipynb

Paper:

https://www.aclweb.org/anthology/D16-1057/

Twitter Specific paper

http://www.marksanderson.org/publications/my_papers/ADC2014.pdf

### Save the Model To Some Format For Scoring

### Stand Up API Service That Has Inputs and Outputs

#### Input: Plain Text

#### Output: Prediction of Sentiment & Probability of Prediction



### Model Deployment

Dockerizing Colab

SSH into Colab


## Discussion

* Scoring is a form of classification

* Supervised model

* Boundary condition classifying an emotion as, "happy" or "not happy" will always yield a result.

* Whether this result can actually be tied to an actual psychologically defined emotion, or rather, behavior is another question.

https://www.semanticscholar.org/paper/Supervised-Term-Weighting-Metrics-for-Sentiment-in-Hamdan-Bellot/e4a9204ee11f5593207c5e262bf26c147620c913/figure/2

* Entire solution found

* Fast ways to solve problems, regular cooking.
* Gourmet Cooking, more creative ways to solve problems.

> What right do we have to say that our model actually describes the desires of our customers?

## Questions

What do the various columns mean?

* _unit_id
* _golden - is it appropriate to consider items categorized as, "golden" as being 100% accurate and items categorized as, "final" as not necessarily known?
* _unit_state
* _trusted_judgments
* _last_judgment_at
* sentiment
* sentiment:confidence
* date
* id
* query
* sentiment_gold
* text
