# Phrase-level movie review sentiment

A naive Bayes sentiment model on Rotten Tomatoes phrases, including a feature that drops determiner words.

## Course

IST 664, Natural Language Processing. The course covers computational techniques for processing written text at each level of linguistic analysis, including tokenization, part of speech tagging, and applications such as sentiment analysis.

## Problem

Each phrase in the instructor-provided Rotten Tomatoes data already has a crowd-sourced sentiment label. The goal was to predict those labels, and to see whether dropping determiner words helped.

## Data

The training set has 156,060 rows and four columns: phrase ID, sentence ID, the phrase, and the sentiment. The test set has the IDs and the phrase only. The sentiment counts were uneven (closer to a bell shape than a flat one). In the notebook I undersampled so each class had the same number of phrases before training. The `corpus/` train and test tables are not in the repo.

## Approach

I built two text features. Feature 0 is the more unusual one: tokens tagged as determiners (POS tag DT) are dropped, then the text is lowercased, stop words are removed, and words are stemmed. Feature 1 skips the determiner step and does the rest.

Each feature went through a count vectorizer as unigrams, bigrams, or both. A multinomial naive Bayes model was trained with 10-fold cross-validation, so there were 60 fitted models in total (two features, three n-gram settings, and ten folds).

## Results

The best average across folds was feature 0 with unigrams and bigrams: accuracy 0.630, precision 0.618, recall 0.630, and F1 0.619. Those ten models were used to score the test phrases. I averaged the class probabilities and wrote the result to `predictions.csv`. That file is not in the repo.

## Tools

Python, NLTK, and scikit-learn (CountVectorizer, MultinomialNB).

## Files

- `code/phrase-sentiment-analysis.ipynb`
- `reports/final-report.docx`
