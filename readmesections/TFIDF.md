[Back to Main](/README.md/)

### TFIDF

After doing some quick research on the topic, TFIDF refers to ["Term Frequency Inverse Document Frequency"](https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089).

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

The bag of words method is not necessarily relevant to our above BERT analysis, however if done previous to our BERT, it could provide some clues on which words to potentially filter out before tokenizing with BERT.

We could also reference this Google CoLab notebook on [TFIDF](https://colab.research.google.com/drive/1h6Jpgcdv2kB07zkcLKFpFM9xsSiZE9pU).

#### Pulling out Cosine similarity

The assignment requested TFIDF, ultimately what is likely being asked for is a demonstration of the relationship between items.  TFIDF is a form of relationship representation, through logarithmic proportion (which, a ratio can be considered a kind of distance or one-dimensional measurement).  The vectors from BERT represent where the words are encoded in the 1024-dimensional hyperspace (1024 for this model uncased_L-24_H-1024_A-16) per [this article](https://towardsdatascience.com/word-embedding-using-bert-in-python-dd5a86c00342). Ultimately those vectors can have cosine distance computed against one another, which would be the analogy to TFIDF in the bag of words model.

To calculate those cosine similarities, we would use the following type of code, though we did not doe this for the assignment:

```
from sklearn.metrics.pairwise import cosine_similarity
cos_lib = cosine_similarity(vectors[1,:],vectors[2,:]) #similarity between #cat and dog

```

[Back to Main](/README.md/)
