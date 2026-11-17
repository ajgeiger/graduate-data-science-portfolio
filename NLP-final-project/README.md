# Phrase-level movie review sentiment

NLP course final project, August 2022.

This is a sentiment model on the Rotten Tomatoes phrase data the instructor provided. Each phrase already has a crowd-sourced sentiment label. The training set has 156,060 rows and four columns: phrase ID, sentence ID, the phrase, and the sentiment. The test set has the IDs and the phrase only.

The report notes that the sentiment counts were uneven (closer to a bell shape than a flat one) and says the labels were left unbalanced. The notebook also includes an undersampling step before training.

## Approach

I built two text features. Feature 0 is the more unusual one: tokens tagged as determiners (POS tag DT) are dropped, then the text is lowercased, stop words are removed, and words are stemmed. Feature 1 skips the determiner step and does the rest.

Each feature went through a count vectorizer as unigrams, bigrams, or both. A multinomial naive Bayes model was trained with 10-fold cross validation, so there were 60 fitted models in total (two features, three n-gram settings, and ten folds).

The best average across folds was feature 0 with unigrams and bigrams: accuracy 0.6193, precision 0.604866, recall 0.619313, and F1 0.607127. Those ten models were used to score the test phrases. I averaged the class probabilities and wrote the result to `predictions.csv`. That file is not in the repo, and the `corpus/` train and test tables are not here either.

## Tools

Python, NLTK, and scikit-learn (CountVectorizer, MultinomialNB).

## Files

- `code/Alex_Geiger_Final_Project.ipynb`
- `reports/Alex_Geiger_Final_Project.docx`
