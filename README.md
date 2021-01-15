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
6. Write up the report within the README file. Throughout the above, take notes on ideas for improvements in the README file and update as we progress.

Any of the above steps may happen at any time, the above is not necessarily a sequential process, but more of a general leaning.

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
* There may have also been a third stage, where a Meta Contributor or perhaps a third user, a Super Contributor, or multiple Super Contributors may have done an an additional analysis of the sentiment column ratings, and entered in an actual sentiment value that they believe to be true, under sentiment_gold. However, this column, sentiment_gold, may have been a, "leftover" column from a completely different Twitter sentiment analysis exercise.


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

Further to this question on the definition of "gold," or "golden," there is an additional column, "sentiment_gold" which contains the values described above. Through research online, it appears that this column may actually have been left over from another sentiment analysis exercise, this link to an, [Intro to Python Twitter Sentiment Guide Document](https://rstudio-pubs-static.s3.amazonaws.com/461021_d910cc02947241669b8c844ff1433a11.html), shows the same column name, but also includes a column for, "sentiment_gold_reason," which includes written reasons for the "sentiment_gold" ratings, making it seem like the "sentiment_gold," is the fully finalized version of the data.

#### Editing Questions Prior to Sending Off

Often in work environments we are faced with coworkers, customers, management, stakeholders, vendors and others who are completely pressed for time and unable to answer all of our questions at length or give large amounts of information.  Our challenge is to not merely collect information, but to do so in a way that builds a good relationship and respects the recipient of our questions and inquiries.

The challenge is of course, not to overburden people with meetings or massively long emails, but rather to be helpful and thoughtful in our communication.

That being said, rather than dumping a large list of questions about every single column in a dataset, I err on the side of asking a few key questions to start off with, with the hope being that long-term, if the project importance is high, we could go through more of a, "product management" process and ask and answer questions at length.  That being said, to formalize my understanding of the data above, I decided to ask a few key questions for this assignment for further clarification:

##### Questions to Send

```
For the purposes of this assignment, can you answer any of these questions to the best of your ability?  If you don't know the answer or don't have time, that's fine you can just let me know that too - this is just an exercise, by my understanding, so this is not super critical and I don't want you to feel like you are not being responsive or respectful to me if you don't really have a solid answer. :

1. Can we consider, the "_golden" column to have the following meaning:	TRUE can be assumed as being 100% accurate, as verified by someone qualified and internal to _Company_ while FALSE cannot.
2. Can we consider, "_trusted_judgments" to mean, a second stage of the creation of the .csv file in which someone went through and looked at the, "sentiment" column and double checked whether the sentiments looked right or not?
3. Or, would it be better to consider the, "sentiment:confidence" to represent some metric of confidence as to whether the "sentiment" column was correct?  Is there a preferred method to, verify or double check sentiment out of these two methods, from the standpoint of _Company_ for this exercise?
4. sentiment_gold is fairly unstructured and seems to have one datapoint from, "text."  Should we accept sentiment to be a, "throw away," or does "sentiment_gold" in fact mean that this is actually the "golden" datapoint for sentiment rather than the, "sentiment" column?

Thank you for your time and consideration in any of the above that you may have time to be able to answer.

```
After asking this set of questions, I got the following answer back, which helped clear things up:

![Stakeholder Input](/assets/images/stakeholderinput.png)

So the basic, default answer was: "You can assign it meaning, or it can be ignored to make things simpler," or perhaps more importantly, "ignore it to make it simpler."

### Compute Set of Engineered Features

### Use regex functions to extract some specific terms

Identification of Regex Functions could be done on specific terms, but if we are looking at this entire project holistically, we can observe that the dataframe columns include all sorts of interesting info, including time series information which could be utilized in different interesting ways in the future. Since this project may possibly include requests for future research, it would be interesting to see how to perhaps, "bulk clean," the entire text column in a way that makes the data more accessible for future project iterations.

Intuitively, Twitter is a fairly well known web element, and likely there are some pre-existing libraries of regex's out there which we may be able to use. A cursory investigation yielded this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).

This documentation might be appropriate for a longer-term, enhanced version of the software that cleans out all possible non-word characters, but starting off with, I did some other quick searches and found a [Stackoverflow answer with a pre-built function](https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression) which already include a wide variety of Python-based Twitter regex extractors. From experience, I understand regex removal work can be fairly tedious, so I decided to make this workable and then move on with the exercise, since I'm attempting to optimize for knowledge demonstration rather than platform construction.

### Compute Word Embeddings (i.e. word2vec, GloVe, BERT)

https://datascience.stackexchange.com/questions/30516/how-does-one-go-about-feature-extraction-for-training-labelled-tweets-for-sentim

#### Researching Steps to Compute Word Embeddings

Basically, several options were suggested by _Company_ as potential ways to get at word embeddings.  Going in any one direction without having much background information on the topic would invite overcomplexity and excess work, so rather than barreling forward at this point, it would be useful to possibly test out and inform myself on what the various types are.

#### Comparative Analysis

| Method                                               | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| word2vec                                             | Method  | Shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors  are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.  This positioning is established by the cosine similarity between the vectors.                                                                                                                                              |
| GloVe                                                | Method  | "Global Vectors," is an unsupervised method which works by mapping words into a meaningful space where the distance between words is related to semantic similarity. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.                                                                                                            |
| GN-GloVe                                             | Method  | Gender Neutral Global Vectors - the same as Global Vectors, but eliminates Gender information from the corpus.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Flair Embeddings                                     | Method  | Flair embeddings trained without any explicit notion of words and thus fundamentally model words as sequences of characters and they are contextualized by their surrounding text, meaning that the same word will have different embeddings depending on its contextual use.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Allen NLP's ELMo                                     | Method  | Character-level tokens are taken as the inputs to a bi-directional LSTM which produces word-level embeddings. Like BERT (but unlike the word embeddings produced by "Bag of Words" approaches, and earlier vector approaches such as Word2Vec and GloVe ), ELMo embeddings are context-sensitive, producing different representations for words that share the same spelling but have different meanings (homonyms) such as "bank" in "river bank" and "bank balance"                                                                                                                                                                                                                                  |
| BERT                                                 | Method  | BERT is not fully understood, as it appears to be proprietary to Google, however it appears to use a Transformer method, as opposed to a RNN method such as LSTM, which is a similar methodology used by GPT. Uses RNN's combined with an attention mechanism, which stores and propagates relevant information down the sequential chain of events within the RNN to the last node. Unlike RNNs, Transformers do not require that the sequential data be processed in order. For example, if the input data is a natural language sentence, the Transformer does not need to process the beginning of it before the end.This allows for greater parallelization and therefore shorter training times. |
| fastText                                             | Library | fastText is a library for learning of word embeddings and text classification created by Facebook's AI Research (FAIR) lab. The model allows one to create an unsupervised learning or supervised learning algorithm for obtaining vector representations for words. The library was generated via Neural Network methods.                                                                                                                                                                                                                                                                                                                                                                             |
| Gensim                                               | Library | Gensim is an open source library. Gensim includes streamed parallelized implementations of fastText, word2vec and doc2vec algorithms, as well as latent semantic analysis (LSA, LSI, SVD), non-negative matrix factorization (NMF), latent Dirichlet allocation (LDA), tf-idf and random projections.                                                                                                                                                                                                                                                                                                                                                                                                  |
| Indra                                                | Library | Indra is an efficient library and service to deliver word-embeddings and semantic relatedness to real-world applications in the domains of machine learning and natural language processing. It offers 60+ pre-build models in 15 languages and several model algorithms and corpora. Indra is powered by spotify-annoy delivering an efficient approximate nearest neighbors  function.                                                                                                                                                                                                                                                                                                               |
| Principal Component Analysis (PCA)                   | Method  | PCA is a form of transform which involves multiple steps, first parameterizing a large vector onto a euclidian space by distance calculation, by mapping all observations on said space as a function of the distance to the center of the space, with a selected distance measurement, typically Cosine distance. Once mapped onto a euclidian space, the observations can then be clustered or a boundary can be created which describes a particular word as a particular grouping, for example, "negative" or "positive."                                                                                                                                                                          |
| T-Distributed Stochastic Neighbour Embedding (t-SNE) | Method  | The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric, this can be changed as appropriate.                 |




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


## Issues or Flagged Items for Future Improvement

### Simpler Improvements (Grunt Coding Work)

* Expanding Regex accuracy for all potential cases.  Essentially, generalizing the Regex according to this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).
* Creating a user prompt that allows a data scientist to select which columns they would like to utilize in the creation of a training or performance measuring system, to be able to compare results from different types of inputs and outputs, since it might be unclear which data is considered golden and which is not.

### Time Consuming, Expensive Improvements

* Comparing predefined, documented measurement methods against each other for performance improvement.
** Essentially, comparing existing tools and methods against each other to ensure that a particular direction makes sense, performance wise, or at least being able to compare algorithms against each other, within reason and without being overly-obsessive about which precise algorithm selected and whether it provides a relatively small percent performance improvement when measuring input to output efficiency vs. balancing other project needs.
* Implementing test-driving development, using, "try," and other standard best practices, perhaps writing tests first and then the actual code itself to ensure viability.
* Getting a better understanding of what system this may be implemented on, what the long-term software architecture may be, and providing for security analysis options and implementation. Some of these may be easy, no-brainers such as using environmental variables tied to a server, but there may also be a full-range of security best practices that could be implemented, depending upon the vulnerability and value of the underlying software in the future.

### More Advanced Improvements

* Gaining a fuller understanding of what customers or stakeholders really meant by their comments. Basically, gaining deeper subject matter expertise on a particular topic, and being able to refine models by specific insights and sentiment quality which requires actually speaking with and creating a human feedback loop between customers and those building the models.
* Gaining a better, contextual understanding of what the IT and security needs of the overall organization are, and creating security standards and protocols for software that we may design to integrate with larger systems.
* Creating a formalized documentation, comment protocol and standard software design practice project document to keep code clean, nice and understandable going forward.

## Citations and Notes:

https://ieeexplore.ieee.org/document/8022667
The results show that the manually indicated tokens combined with a Decision Tree classifier outperform any other feature set-classification algorithm combination. The manually annotated dataset that was used in our experiments is publicly available for anyone who wishes to use it.

http://mpqa.cs.pitt.edu/
UPitt Subjectivity Lexicon
