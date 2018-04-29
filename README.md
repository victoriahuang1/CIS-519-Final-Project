# CIS-519-Final-Project

## Goals
* Predict language family
* Predict specific language of text
* How does the classifier's performance vary across different corpora?
* Predict language of words in text with code-switching
* Feature analysis (what features work well?)


## Project Setup
* Test and train data are in the data/ folder
* Source files are in the root directory
* The commands provided below are intended to be used in the root directory of the project


## Analyzing performance

* We will be using accuracy as our measure for performance
* We have provided a script (score.py) to compute the accuracy. To use it, provide the path to the
predictions file as the first argument, and provide the path to the gold labels as the second
argument. See example of how to run the script below:

$ python3.6 score.py pred_file

$ Accuracy: 0.5


## Simple Baseline

* Predicts labels based on probability distribution defined by the label proportions
* All further work should be able to beat this baseline
* TODO: explain how to run


## Language Model Baseline

* We have a separate model for each language. Each model is trained on one particular language and
gives a score of how confident it is that the given document is written in that language.
* We take the prediction of the model with the highest score (One-vs-All with Winner Takes All)
* TODO: explain how to run


## Extensions

We tried several additional ideas to build off of our language model baseline:
* Incorporating more features: TODO: list features
* Boosting with language models as features in a linear model (SVM) TODO: change?
* Using the naive bayes assumption for determining the probability of code-switching as opposed to
predicting the the language of each word separately
