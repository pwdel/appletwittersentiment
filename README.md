# Apple Twitter Sentiment Analysis

Apple Twitter Sentiment Analysis Project

1. [Overview](/readmesections/Overview.md) - Project Objective, My Background, My Plan of Attack

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

#### Researching Steps to Compute Word Embeddings

[General instructions on how to create word embeddings](https://datascience.stackexchange.com/questions/30516/how-does-one-go-about-feature-extraction-for-training-labelled-tweets-for-sentim) which can roughly be described as follows:

Word embeddings are essentially different methods of ascribing value to words based upon many various types of algorithms. Some of these algorithms are rudimentary, well understood, and look at the route quantity of words to ascribe importance. Other algorithms are understood and assign weighting, either through decision trees, neural networks or some other methodology. Still other algorithms are actually proprietary black boxes, such as BERT, which has pre-defined inputs and outputs, and while we may not be able to completely reverse-engineer these algorithms yet, they have been ascribed importance by highly influential computing journals and peer reviewed, and so are more or less accepted as gold standards by a community, at least for the purposes of linguistic usage. The philosophy behind being able to use these types of algorithms for more hard empirical science may be suspect, but in the commercial NLP world, often it may be more advantageous to use something based upon reputation and ease of use to go faster to market, rather than worry about deep, costly, empirical research.

After word embeddings are computed into a vector format, they can be handed off to a learning algorithm to use to make predictions.

Basically, several options were suggested by _Company_ as potential ways to get at word embeddings.  Going in any one direction without having much background information on the topic would invite overcomplexity and excess work, so rather than barreling forward at this point, it would be useful to possibly test out and inform myself on what the various types are.

First off, a general definition to word embeddings

> Word embeddings refers to representing words or sentences in a numerical vector form or word embedding opens up the gates to various potential applications. This functionality of encoding words into vectors is a powerful tool for NLP tasks such as calculating semantic similarity between words with which one can build a semantic search engine.

#### Comparative Analysis

| Method                                               | Type    | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|------------------------------------------------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| word2vec                                             | Method  | Shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. Word2vec takes as its input a large corpus of text and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. Word vectors  are positioned in the vector space such that words that share common contexts in the corpus are located close to one another in the space.  This positioning is established by the cosine similarity between the vectors.                                                                                                                                              |
| GloVe                                                | Method  | "Global Vectors," is an unsupervised method which works by mapping words into a meaningful space where the distance between words is related to semantic similarity. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space. The advantage of GloVe is that, unlike Word2vec, GloVe does not rely just on local statistics (local context information of words), but incorporates global statistics (word co-occurrence) to obtain word vectors.                                                                                                            |
| GN-GloVe                                             | Method  | Gender Neutral Global Vectors - the same as Global Vectors, but eliminates Gender information from the corpus.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| Flair Embeddings                                     | Method  | Flair embeddings trained without any explicit notion of words and thus fundamentally model words as sequences of characters and they are contextualized by their surrounding text, meaning that the same word will have different embeddings depending on its contextual use.                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Allen NLP's ELMo                                     | Method  | Character-level tokens are taken as the inputs to a bi-directional LSTM which produces word-level embeddings. Like BERT (but unlike the word embeddings produced by "Bag of Words" approaches, and earlier vector approaches such as Word2Vec and GloVe ), ELMo embeddings are context-sensitive, producing different representations for words that share the same spelling but have different meanings (homonyms) such as "bank" in "river bank" and "bank balance"                                                                                                                                                                                                                                  |
| BERT                                                 | Method  | BERT is not fully understood, as it appears to be proprietary to Google, however it appears to use a Transformer method, as opposed to a RNN method such as LSTM, which is a similar methodology used by GPT. BERT relies on massive compute for pre-training ( 4 days on 4 to 16 Cloud TPUs; pre-training on 8 GPUs would take 40–70 days i.e. is not feasible. BERT fine tuning tasks also require huge amounts of processing power, which makes it less attractive and practical for all but very specific tasks. Typical uses would be fine tuning BERT for a particular task or for feature extraction. Uses RNN's combined with an attention mechanism, which stores and propagates relevant information down the sequential chain of events within the RNN to the last node. Unlike RNNs, Transformers do not require that the sequential data be processed in order. For example, if the input data is a natural language sentence, the Transformer does not need to process the beginning of it before the end.This allows for greater parallelization and therefore shorter training times. |
| fastText                                             | Library | fastText is a library for learning of word embeddings and text classification created by Facebook's AI Research (FAIR) lab. The model allows one to create an unsupervised learning or supervised learning algorithm for obtaining vector representations for words. The library was generated via Neural Network methods.                                                                                                                                                                                                                                                                                                                                                                             |
| Gensim                                               | Library | Gensim is an open source library. Gensim includes streamed parallelized implementations of fastText, word2vec and doc2vec algorithms, as well as latent semantic analysis (LSA, LSI, SVD), non-negative matrix factorization (NMF), latent Dirichlet allocation (LDA), tf-idf and random projections.                                                                                                                                                                                                                                                                                                                                                                                                  |
| Indra                                                | Library | Indra is an efficient library and service to deliver word-embeddings and semantic relatedness to real-world applications in the domains of machine learning and natural language processing. It offers 60+ pre-build models in 15 languages and several model algorithms and corpora. Indra is powered by spotify-annoy delivering an efficient approximate nearest neighbors  function.                                                                                                                                                                                                                                                                                                               |
| Principal Component Analysis (PCA)                   | Method  | PCA is a form of transform which involves multiple steps, first parameterizing a large vector onto a euclidian space by distance calculation, by mapping all observations on said space as a function of the distance to the center of the space, with a selected distance measurement, typically Cosine distance. Once mapped onto a euclidian space, the observations can then be clustered or a boundary can be created which describes a particular word as a particular grouping, for example, "negative" or "positive."                                                                                                                                                                          |
| T-Distributed Stochastic Neighbour Embedding (t-SNE) | Method  | The t-SNE algorithm comprises two main stages. First, t-SNE constructs a probability distribution over pairs of high-dimensional objects in such a way that similar objects are assigned a higher probability while dissimilar points are assigned a lower probability. Second, t-SNE defines a similar probability distribution over the points in the low-dimensional map, and it minimizes the Kullback–Leibler divergence (KL divergence) between the two distributions with respect to the locations of the points in the map. While the original algorithm uses the Euclidean distance between objects as the base of its similarity metric, this can be changed as appropriate.                 |


#### Summary of Comparative Analysis

Basically, the above table discusses some of the abstract mathematics between various methodologies and libraries within Natural Language Processing.  Overall:

* BERT appears to be thought of as the most, "accurate," and modern model.  Whereas in the past, LSTM may have been more popular and while many various tutorials exist using LSTM for sentiment analysis online, BERT is thought to take into account some of the weaknesses of RNN-only models, by adding an, "attention," layer which computes relevance of various terms.
* Whereas Word2Vec and Glove would produce different
* Note, I did find existing CoLab notebooks for [word2vec](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb), [GloVe](https://colab.research.google.com/github/mdda/deep-learning-workshop/blob/master/notebooks/5-RNN/3-Text-Corpus-and-Embeddings.ipynb), and [BERT](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb), as well as [TFIDF](https://colab.research.google.com/drive/1h6Jpgcdv2kB07zkcLKFpFM9xsSiZE9pU).


### Bulding Our Own BERT Code for Calculating Embeddings on Training Data

We found and reviewed the following articles on BERT embeddings:

* [Using BERT in Python](https://towardsdatascience.com/word-embedding-using-bert-in-python-dd5a86c00342)
* [Step by Step Implementation of Bert for Text Categorization](https://medium.com/analytics-vidhya/step-by-step-implementation-of-bert-for-text-categorization-task-aba80417bd84)
* [Illustrated Guide to BERT](http://jalammar.github.io/illustrated-bert/)

...as well as the CoLab notbook linked above.

#### Reduction of Data to Golden, Usable Training Dataset

Note: .size() takes into account NaN values, while .count() only counts numbers.

* Count of original dataset, sentimentmix_df is:  3886
* Count of extracted goldens, sentimentmixgolden_df is:  103
* Count of non-neutral ratings, sentimentmatrixgoldennonneutral_df is:  58
* Count of relevant sentimentmatrixgoldennonneutralrelevant_df size is:  57

##### Side Discussion: Is this a Sufficient Dataset Size to Make an Accurate Global Prediction?

The above analysis shows that out of the entire original dataset including 3886 sentiment datapoints, realistically there are only 57 golden sentiment datapoints on which we can make a prediction.

Is this enough?  Given our domain expertise at this point, we can't say. One thing I like to point out when people ask about sufficient data used to create a prediction, is that it important to note that there is no mathematically certifiable way to calculate whether a prediction will be sufficient for any prediction, globally. Often data scientists and scientists in general confuse the concept of, "confidence interval" to mean that a sufficient amount of data must be captured to have, "confidence," with a prediction.  This is not a proper interpretation of, "confidence interval," and more information about the proper interpretation of confidence interval can be found in some of my other writings.

Ultimately, there has to be enough domain expertise to understand and be able to know what a sufficient amount of data will be for a global prediction. While many individuals have put together tutorials on this Apple Twitter dataset, and have demonstrated accurate algorithms within the scope of the project, this is merely fitting the data to the problem itself, basically forcing a fit, which is not true prediction. Anyone can sit and optimize code and force a fit, few can combine expertise and math to optimize real-world results.

As far as our own layperson expertise goes, we can look at our fully cleaned sentiment data, since it only encompasses 57 points, and verify manually fairly quickly whether the data appears to at least be intuitively, "fitting," and not overlapping. This visual inspection showed that what is shown as being a 0 is indeed a negative sentiment toward Apple, and what is shown as being a 1 is a positive sentiment.

With some cursory reading on the topic, [this article mentions](https://towardsdatascience.com/latent-semantic-analysis-sentiment-classification-with-python-5f657346f6a3) that an accurate model would require between 10,000 and 30,000 features to train a decently accurate model using the TFIDF method.

![cleaned sentiment analysis](/assets/images/cleanedsentiment.png)

#### Inputting Dataframe into BERT Tokenizer

According to [this article](https://albertauyeung.github.io/2020/06/19/bert-tokenization.html) tokenization can either be done directly within BERT, or with another outside package called, "transformers."  I attempted to run things via BERT, but ran into some errors, so quickly pivoted to transformers and this worked as shown below.

I found this Github gist which shows a [method of how to prepare a Pandas dataseries for BERT Transformers tokenization](https://gist.github.com/akshay-3apr/0a345b4f416051b3676aae31a14bbde2).

Utilizing that as a framework, I wrote my own version using our selected dataset from above, but included the attention mask option, since this was the original interesting thing about BERT. I was unable to figure out how to pull out the attention mask, so for now I'm leaving this as an unknown.

```

## create label and sentence list
sentences = transformerready_df.text.values

#check distribution of data based on labels
print("Distribution of data based on labels: ",transformerready_df.sentiment.value_counts())

# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway.
# In the original paper, the authors used a length of 512.
MAX_LEN = 512

## Import BERT tokenizer, that is used to convert our text into tokens that corresponds to BERT library
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
# ask tokenizer to encode words
input_ids = [tokenizer.encode(sent,truncation=True,add_special_tokens=True,max_length=MAX_LEN,pad_to_max_length=True) for sent in sentences]
labels = transformerready_df.sentiment.values

print("Actual sentence before tokenization, (1): ",sentences[1])
print("Encoded Input from dataset (1): ",input_ids[1])

print("Actual sentence before tokenization, (2): ",sentences[2])
print("Encoded Input from dataset (2): ",input_ids[2])

```

This yielded tokenized versions of the sentences in question under 'text'.

### TFIDF

After doing some quick research on the topic, TFIDF refers to []"Term Frequency Inverse Document Frequency"](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089).

TF is simply a calculation of the term frequency in a document, whereas IDF is the inverse of the document frequency which measures the informativeness of term t.  A generalized formatting of the mathematics behind TFIDF is shown below:

```
tf(t,d) = count of t in d / number of words in d

df(t) = occurrence of t in documents

Inverse of that document set N divided by occurrence of t in documents

idf(t) = N/df

During the query time, when a word which is not in vocab occurs, the df will be 0. As we cannot divide by 0, we smoothen the value by adding 1 to the denominator.  So basically, the size of the text is logarithmicly proportional to the occurrence of the term in all texts. The size of the total text in all documents matters, "more" weighted against the number of occurrences of a word. This gets normalized against the number of times a term occurs in a given document.

So essentially, the word, "love" may occur a lot in one Sonnet by Shakespeare, and a Sonnet is relatively small, so it gains a high relative importance. However if you consider all of the plays ever written by Shakespeare, "love" may occur very frequently, but the importance shrinks logarithmically. Basically terms which are not super common across an entire body of work and only show up on a page here and there get, "smoothed out," whereas domain-specific terms that occur again and again across a large amount of text percolate out to the top.

idf(t) = log(N/(df + 1))

tf-idf(t, d) = tf(t, d) * log(N/(df + 1))

Ultimately,

TF-IDF = Term Frequency (TF) * Inverse Document Frequency (IDF)

```

Terminologies:

* t — term (word)
* d — document (set of words)
* N — count of corpus
* corpus — the total document set

Of course when calculating TFIDF, there will be certain pre-data cleaning considerations, such as removing special characters and stop-words. For the purposes of this exercise, I am opting not to remove stop words, since the overall analysis that we are using leverages BERT, and it would be a guessing game to know which words BERT considers to be stop words.  We could have removed these stop words manually earlier in the project if we wished, but for simplification purposes, we can ignore stop words and make the assumption that BERT will handle them, even if they won't.

While the above article is very detailed and interesting, and provides a good reference, for the purposes of this assignment we just want a general count of TFIDF, which can be established by the following code, from this [Stackoverflow](https://stackoverflow.com/questions/37593293/how-to-get-tfidf-with-pandas-dataframe), leveraging sklearn.

```
from sklearn.feature_extraction.text import TfidfVectorizer
v = TfidfVectorizer()
x = v.fit_transform(df['sent'])
```

From here we can visualize our TFDIF using PCA, per this [Stackoverflow inquiry](https://stackoverflow.com/questions/28160335/plot-a-document-tfidf-2d-graph).  

> When you use Bag of Words, each of your sentences gets represented in a high dimensional space of length equal to the vocabulary. If you want to represent this in 2D you need to reduce the dimension, for example using PCA with two components:

Upon working out our vector with the code above, we get a

```
<57x427 sparse matrix of type '<class 'numpy.float64'>' with 710 stored elements in Compressed Sparse Row format>
```

With head values:

```
(0, 302)	0.4857309068059386
  (0, 103)	0.4086390353893864
  (0, 346)	0.4857309068059386
  (0, 25)	0.4086390353893864
```

This is a sparse matrix, visualizing in a plot looks like the following:

![sparce matrix](/assets/images/sparcematrix.png)

The bag of words method is not necessarily relevant to our above BERT analysis, however if done previous to our BERT, it could provide some clues on which words to poentially filter out before tokenizing with BERT.

We could also reference this Google CoLab notebook on [TFIDF](https://colab.research.google.com/drive/1h6Jpgcdv2kB07zkcLKFpFM9xsSiZE9pU).

#### Pulling out Cosine similarity

The assignment requested TFIDF, ultimately what is likely being asked for is a demonstration of the relationship between items.  TFIDF is a form of relationship representation, through logarithmic proportion (which, a ratio can be considered a kind of distance or one-dimensional measurement).  The vectors from BERT represent where the words are encoded in the 1024-dimensional hyperspace (1024 for this model uncased_L-24_H-1024_A-16) per [this article](https://towardsdatascience.com/word-embedding-using-bert-in-python-dd5a86c00342). Ultimately those vectors can have cosine distance computed against one another, which would be the analogy to TFIDF in the bag of words model.

To calculate those cosine similarities, we would use the following type of code, though we did not doe this for the assignment:

```
from sklearn.metrics.pairwise import cosine_similarity
cos_lib = cosine_similarity(vectors[1,:],vectors[2,:]) #similarity between #cat and dog

```

#### Fine Tuning Bert Models

Training a BERT model is known as, "Fine Tuning," since technically BERT is already trained on a massive Google dataset, and we as users are really just modifying a highly expensive training dataset output for our own use. Fine Tuning BERT models, which is really what is needed to come up with a superior result, is considerably complicated and so I will revert back to classical logistic regression and tokenization methods for the remainder of this assignment, though I understand it is not optimal in terms of accuracy.

### Train Classifier to Predict the 'Sentiment' Class

#### Further Research on Topic

* Through researching dataset definitions in an above section, I was actually able to find an entire solution set to this assignment [at this blog here](https://harrisonjansma.com/apple), including tokenization, which already claimed an 89% accuracy using linear model using Logistic Regression, claiming a performance lead over a linear Support Vector Machine model.

#### Above

We used the above linear regression model to extract features from our previously generated, small golden dataset.

```
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

s1 = transformerready_df['text']
s2 = sentimentmatrixgoldennonneutralrelevant_df['sentiment:confidence']

X = pd.concat([s1,s2], axis=1)
y = transformerready_df['sentiment']

#splitting data for cross validation of model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,shuffle=False)

#Keeping the assignment confidence for later
X_train_conf, X_test_conf = X_train['sentiment:confidence'], X_test['sentiment:confidence']
X_train, X_test = X_train['text'], X_test['text']

#saving to csv
X_train.to_csv('train_clean.csv')
X_test.to_csv('test_clean.csv')
y_train.to_csv('y_train.csv')
y_test.to_csv('y_test.csv')

print(X_train[:5])

```

We then built a linear model with sklearn LogisticRegression, as shown below.

```
from sklearn.linear_model import LogisticRegressionCV

logr = LogisticRegressionCV()
logr.fit(X_train_tfidf, y_train)
y_pred_logr = logr.predict(X_test_tfidf)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

clf = SVC(class_weight = 'balanced')
pipe = Pipeline([('classifier', clf)])
fit_params = {'classifier__kernel':['rbf', 'linear', 'poly'],
          'classifier__degree':[2, 3, 4],
          'classifier__C':[0.01, 0.1, 1, 10]}

gs = GridSearchCV(pipe, fit_params, cv = 10, return_train_score = True)
gs.fit(X_train_tfidf, y_train)

print('Best performing classifier parameters (score {}):\n{}'.format(gs.best_score_, gs.best_params_))

pipe.set_params(classifier__degree = gs.best_params_['classifier__degree'],
                classifier__kernel = gs.best_params_['classifier__kernel'],
               classifier__C = gs.best_params_['classifier__C'])
pipe.fit(X_train_tfidf, y_train)
y_pred = pipe.predict(X_test_tfidf)
```

![Most Important Words Analysis](/assets/images/positivenegativecounts.png)

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

* Linear Regression Assumptions

* What is the sentiment really positive about?

* Even supposudly golden samples can have irregularities, after all this is all originally human input.  Therefore, additional cleaning using uniques needs to be done.

* Size of data - as we reduced down from golden samples and non-neutral samples

Interesting Learnings within this section:

One interesting part of NLP that I learned through this assignment is that within this specific domain, there are already thousands of pre-trained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation, in 100+ human languages. This domain has a wide amount of research work pre-completed, and these pretrained models can be found at [Python Transformers](https://pypi.org/project/transformers/).  Transformers integrates with PyTorch and Tensorflow and has ready-made API's.

## Issues or Flagged Items for Future Improvement

### Simpler Improvements (Grunt Coding Work)

* Expanding Regex accuracy for all potential cases.  Essentially, generalizing the Regex according to this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).
* Creating a user prompt that allows a data scientist to select which columns they would like to utilize in the creation of a training or performance measuring system, to be able to compare results from different types of inputs and outputs, since it might be unclear which data is considered golden and which is not.
* Creating extensive lists of stop words, either based upon manual flagging, or gathering them from online and adding them into our models. Certain stop words such as, "a" and "the" might be no-brainers, but this could turn into a more complex project as more stop terms get added.
* Add confusion matrix, Distributions of Predictions and ROC Curve.

### Time Consuming, Expensive Improvements

* Comparing predefined, documented measurement methods against each other for performance improvement.
** Essentially, comparing existing tools and methods against each other to ensure that a particular direction makes sense, performance wise, or at least being able to compare algorithms against each other, within reason and without being overly-obsessive about which precise algorithm selected and whether it provides a relatively small percent performance improvement when measuring input to output efficiency vs. balancing other project needs.
* Implementing test-driving development, using, "try," and other standard best practices, perhaps writing tests first and then the actual code itself to ensure viability.
* Getting a better understanding of what system this may be implemented on, what the long-term software architecture may be, and providing for security analysis options and implementation. Some of these may be easy, no-brainers such as using environmental variables tied to a server, but there may also be a full-range of security best practices that could be implemented, depending upon the vulnerability and value of the underlying software in the future.
* Within a measurement method, compare optional settings against one another and create a decision tree or other uber-algorithm which finds optimal results. For example, with the BERT example used, one could optionally activate the attention mask option or not and see if this makes a difference.
* Fine tuning a BERT model for this application based upon the vectors we extracted.

### More Advanced Improvements

* Gaining a fuller understanding of what customers or stakeholders really meant by their comments. Basically, gaining deeper subject matter expertise on a particular topic, and being able to refine models by specific insights and sentiment quality which requires actually speaking with and creating a human feedback loop between customers and those building the models.
* Gaining a better, contextual understanding of what the IT and security needs of the overall organization are, and creating security standards and protocols for software that we may design to integrate with larger systems.
* Creating a formalized documentation, comment protocol and standard software design practice project document to keep code clean, nice and understandable going forward.

## Citations and Notes:

https://ieeexplore.ieee.org/document/8022667
The results show that the manually indicated tokens combined with a Decision Tree classifier outperform any other feature set-classification algorithm combination. The manually annotated dataset that was used in our experiments is publicly available for anyone who wishes to use it.

http://mpqa.cs.pitt.edu/
UPitt Subjectivity Lexicon

http://www.marksanderson.org/publications/my_papers/ADC2014.pdf

https://www.cs.uic.edu/~liub/FBS/NLP-handbook-sentiment-analysis.pdf


https://github.com/apmoore1/SentiLexTutorial

https://github.com/apmoore1/SentiLexTutorial/blob/master/Tutorial.ipynb

Paper:

https://www.aclweb.org/anthology/D16-1057/

Twitter Specific paper

http://www.marksanderson.org/publications/my_papers/ADC2014.pdf
