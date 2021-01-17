[Back to Main](/README.md/)

### Compute Set of Engineered Features

### Use regex functions to extract some specific terms

Identification of Regex Functions could be done on specific terms, but if we are looking at this entire project holistically, we can observe that the dataframe columns include all sorts of interesting info, including time series information which could be utilized in different interesting ways in the future. Since this project may possibly include requests for future research, it would be interesting to see how to perhaps, "bulk clean," the entire text column in a way that makes the data more accessible for future project iterations.

Intuitively, Twitter is a fairly well known web element, and likely there are some pre-existing libraries of regex's out there which we may be able to use. A cursory investigation yielded this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).

This documentation might be appropriate for a longer-term, enhanced version of the software that cleans out all possible non-word characters, but starting off with, I did some other quick searches and found a [Stackoverflow answer with a pre-built function](https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression) which already include a wide variety of Python-based Twitter regex extractors. From experience, I understand regex removal work can be fairly tedious, so I decided to make this workable and then move on with the exercise, since I'm attempting to optimize for knowledge demonstration rather than platform construction.

### Reduction of Data to Golden, Usable Training Dataset

Note: .size() takes into account NaN values, while .count() only counts numbers.

* Count of original dataset, sentimentmix_df is:  3886
* Count of extracted goldens, sentimentmixgolden_df is:  103
* Count of non-neutral ratings, sentimentmatrixgoldennonneutral_df is:  58
* Count of relevant sentimentmatrixgoldennonneutralrelevant_df size is:  57

#### Side Discussion: Is this a Sufficient Dataset Size to Make an Accurate Global Prediction?

The above analysis shows that out of the entire original dataset including 3886 sentiment datapoints, realistically there are only 57 golden sentiment datapoints on which we can make a prediction.

Is this enough?  Given our domain expertise at this point, we can't say. One thing I like to point out when people ask about sufficient data used to create a prediction, is that it important to note that there is no mathematically certifiable way to calculate whether a prediction will be sufficient for any prediction, globally. Often data scientists and scientists in general confuse the concept of, "confidence interval" to mean that a sufficient amount of data must be captured to have, "confidence," with a prediction.  This is not a proper interpretation of, "confidence interval," and more information about the proper interpretation of confidence interval can be found in some of my other writings.

Ultimately, there has to be enough domain expertise to understand and be able to know what a sufficient amount of data will be for a global prediction. While many individuals have put together tutorials on this Apple Twitter dataset, and have demonstrated accurate algorithms within the scope of the project, this is merely fitting the data to the problem itself, basically forcing a fit, which is not true prediction. Anyone can sit and optimize code and force a fit, few can combine expertise and math to optimize real-world results.

As far as our own layperson expertise goes, we can look at our fully cleaned sentiment data, since it only encompasses 57 points, and verify manually fairly quickly whether the data appears to at least be intuitively, "fitting," and not overlapping. This visual inspection showed that what is shown as being a 0 is indeed a negative sentiment toward Apple, and what is shown as being a 1 is a positive sentiment.

With some cursory reading on the topic, [this article mentions](https://towardsdatascience.com/latent-semantic-analysis-sentiment-classification-with-python-5f657346f6a3) that an accurate model would require between 10,000 and 30,000 features to train a decently accurate model using the TFIDF method.

![cleaned sentiment analysis](/assets/images/cleanedsentiment.png)

[Back to Main](/README.md/)
