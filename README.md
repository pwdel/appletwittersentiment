# Apple Twitter Sentiment Analysis

Apple Twitter Sentiment Analysis Project

1. [Overview](/readmesections/Overview.md) - Project Objective, My Background, My Plan of Attack
2. [Understanding the Data](/readmesections/Data.md) - Researching the dataset, questioning column validity, organizational and stakeholder considerations.
3. [Preprocessing](/readmesections/Preprocessing.md) - Golden filtering, regex, discussion on whether we have sufficient data to make an accurate prediction.
4. [Word Embeddings - BERT](/readmesections/WordEmbeddings.md) - Evaluating different methods. Creating word embeddings with BERT and deciding to abandon BERT due to implementation time requirements for the purposes of this assignment.
5. [TFIDF](/readmsesections/TFIDF.md) - Discussion and calculation.
6. [Train Model Using Bag of Words Method](/readmesections/BagofWords.md) - Training and evaluating model.
7. [Scoring](/readmesections/Scoring.md) - Saving the trained model for usage in an API.
8. [API](/readmsesections/API.md) -



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
