# Apple Twitter Sentiment Analysis

Apple Twitter Sentiment Analysis Project

## Overall Narrative of Project

The following table provides links to readme files narrating various aspects of the project and explaining my thought process and rationale as I worked through everything.

| Section                                                                | Description                                                                                                                                                                |
|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Overview](/readmesections/Overview.md)                                | Project Objective, My Background, My Plan of Attack                                                                                                                        |
| [Understanding the Data](/readmesections/Data.md)                      | Researching the dataset, questioning column validity, organizational and stakeholder considerations.                                                                       |
| [Preprocessing](/readmesections/Preprocessing.md)                      | Golden filtering, regex, discussion on whether we have sufficient data to make an accurate prediction.                                                                     |
| [Word Embeddings - BERT](/readmesections/WordEmbeddings.md)            | Evaluating different methods. Creating word embeddings with BERT and deciding to abandon BERT due to implementation time requirements for the purposes of this assignment. |
| [TFIDF](/readmesections/TFIDF.md)                                     | Discussion and calculation.                                                                                                                                                |
| [Train Model Using Bag of Words Method](/readmesections/BagofWords.md) | Training and evaluating model.                                                                                                                                             |
| [Scoring](/readmesections/Scoring.md)                                  | Saving the trained model for usage in an API.                                                                                                                              |
| [API](/readmesections/API.md)                                         | Pseudocode, Working with Saved Machine Learning Model, Input and Output Functions, Jsonification                                                                           |
| [Discussion](/readmesections/Discussion.md)                            | Various points of discussion on the assignment.                                                                                                                            |
| [Future Improvement](/readmesections/FutureImprovement.md)             | Simpler improvements, more expensive improvements and advanced improvement ideas.                                                                                          |

## Notebooks and Working

* [Google Drive Folder Used](https://drive.google.com/drive/folders/1WicGkBotOouPvv4pwAk1Frfj7xFOwKG4?usp=sharing)
* [Filtering and Training Colab Notebook](https://colab.research.google.com/drive/1a9ZtMX4TGZmAm_ys1MmKQieoCmPG42V1?usp=sharing)
* [App Building Colab Notebook](https://colab.research.google.com/drive/1OjLswUQWPp5jHD40PC93rNoBZqhuO4Hu?usp=sharing)

## Software Architecture

![Software Architecture](/assets/images/sw-arch.png)

## Instructions for Getting App going

Update 19 Jan 2020

<hr>

| Status | Message                                                                                        |
|--------|------------------------------------------------------------------------------------------------|
| ❌      | Warning!  This app does not work.                                                              |
| ➡️      | I Was unable to get Docker going on MacOS Sierra, Macbook Air. Not compatible.                 |
| ➡️      | I can potentially get this Dockerized using some cloud service, I would need to research that. |

1. Ensure using Python 3.9.  
2. Download main.py and requirements.txt from github repo.
3. Setup virtualenv and set default python version on your local machine, or remember to always run everything with python3.
4. In your terminal: $ python3 main.py
5. Follow app instructions.
