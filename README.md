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

$ python3.6 simple_baseline.py pred_file

Alternatively, you can just run run_simple.sh to get both the predictions and the score.

$ sh run_simple.sh 1    # to predict languages

$ sh run_simple.sh 2    # to predict language families


## Language Model Baseline

* We have a separate model for each language. Each model is trained on one particular language and
gives a score of how confident it is that the given document is written in that language.
* We take the prediction of the model with the highest score (One-vs-All with Winner Takes All)

To run, use the run_language_model.sh script:

$ sh run_language_model.sh 1 3  # to predict languages, using n-grams with a history of the last 3 characters

$ sh run_language_model.sh 2 3  # to predict languages families, using n-grams with a history of the last 3 characters


## Extensions

We tried several additional ideas to build off of our language model baseline:
* Predicting the language of each word in the document separately and making the label the most predicted label in the
document (1)
* Using language models as features in a linear model (SVM) (2)
* Ignoring ambiguous words (i.e. 123 can be in any language) in training and testing
* Incorporating more features: whether all characters are ASCII, average length of words
* Making an ensemble method with different n-gram language models, where the most voted label wins (3)

To run the extensions:

$ sh run_improved.sh 1 1    # to predict languages using extension (3)

$ sh run_improved.sh 2 1    # to predict language families using extension (3)

$ sh run_improved.sh 1 0    # to predict languages using extension (2)

$ sh run_improved.sh 2 0    # to predict languages using extension (2)
