[Back to Main](/README.md/)

## Issues or Flagged Items for Future Improvement

### Simpler Improvements (Grunt Coding Work)

* Expanding Regex accuracy for all potential cases.  Essentially, generalizing the Regex according to this [Ruby Gem documentation](https://www.rubydoc.info/gems/twitter-text/1.13.0/Twitter/Regex).
* Creating a user prompt that allows a data scientist to select which columns they would like to utilize in the creation of a training or performance measuring system, to be able to compare results from different types of inputs and outputs, since it might be unclear which data is considered golden and which is not.
* Creating extensive lists of stop words, either based upon manual flagging, or gathering them from online and adding them into our models. Certain stop words such as, "a" and "the" might be no-brainers, but this could turn into a more complex project as more stop terms get added.
* Add confusion matrix, Distributions of Predictions and ROC Curve.
* Improve CSV upload/download process using proper utf-8 encoding.
* Within our bag of words analysis, we only did a vectorize_fit method with 20% of the available golden data.  If we had used a larger training set, given that our golden data was so limited, perhaps we could have gotten a better fit - however there is of course the risk of over-fitting.

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

<hr>

* Ultimately, the holy grail of any machine learning system is the creation of a feedback loop which expands a system's capability to improve its performance. This is something I have written about in a blog post and a [paper on clustering for the improvement of marketing funnels which can be found here](https://www.patdel.com/video-views-marketing-funnels/). In the context of Semantic Analysis, this would arguably include creating a way to continuously expand the vocabulary of a system, by 1. Introducing new vocabulary into a database or, "corpus of text." 2. Continuously improving the number of golden samples in that database by introducing a system for human review of each incoming Tweet. 3. Updating a learning model based upon filtered, golden samples. 4. Scoring that updated model, and introducing the newly scored model into the API.

Overall, this would form a feedback loop, as I have written about previously in the above-linked paper.

![feedback Loop](/assets/images/feedbackloop.png)

Also noteable as discussed in the section on, [Bag of Words Model Training](/readmesections/BagofWords.md), we had found that the term, "apple" was considered negative, while, "appl," the stock term, was considered positive, which possibly hints at some sort of stock pumping going on.  So when looking at the, "human sentiment gauge," layer, mentioned in steps 2 and 3 above, it would be critical to ensure subject matter expertise around the topic at hand to be able to gauge, "true sentiment," vs. "self-optimized sentiment."

This is different than, "overfitting," which generally takes the form of making an overly complex model to explain idiosyncrasies in the data under study. Overfitting deals with the algorithmic complexity, while gauging true sentiment is more of a soft-skill, basically relying on human awareness to assist downstream quantification.

[Back to Main](/README.md/)
