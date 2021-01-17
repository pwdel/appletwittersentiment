[Back to Main](/README.md/)

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

[Back to Main](/README.md/)
