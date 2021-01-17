[Back to Main](/README.md/)

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

[Back to Main](/README.md/)
