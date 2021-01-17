[Back to Main](/README.md/)

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

#### Fine Tuning Bert Models

Training a BERT model is known as, "Fine Tuning," since technically BERT is already trained on a massive Google dataset, and we as users are really just modifying a highly expensive training dataset output for our own use. Fine Tuning BERT models, which is really what is needed to come up with a superior result, is considerably complicated and so I will revert back to classical logistic regression and tokenization methods for the remainder of this assignment, though I understand it is not optimal in terms of accuracy.

[Back to Main](/README.md/)
