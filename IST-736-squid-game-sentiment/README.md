# Squid Game sentiment

IST 736, Text Mining.

Team: Alex Geiger, Kyle Walter, Eric Arnold, and John Hoyle.

The setup was a client brief: Netflix had picked up Squid Game and wanted a read on public sentiment before committing to another season. We trained sentiment models on IMDb reviews, applied them to tweets, and used LDA to see what people talked about.

The notebook is [`code/Final_Project_Alex_Geiger_code.ipynb`](code/Final_Project_Alex_Geiger_code.ipynb). A knitted report is in [`reports/Final_Project_Alex_Geiger_final_draft.html`](reports/Final_Project_Alex_Geiger_final_draft.html).

## Data

- 1,186 IMDb user reviews (Kaggle), with star ratings used as labels
- More than 80,000 tweets about the show (also from Kaggle)

IMDb scores were heavily skewed high. Oversampling and undersampling both had problems, so we binned stars into five labels: very negative (1-5), negative (6-7), neutral (8), positive (9), and very positive (10).

## Models

Reviews were cleaned (stop words, stemming, and so on) and a multinomial naive Bayes model was tuned on those features. The setting that worked best used the `lem_review_rm_sw_stem` text and an alpha of 0.93. With cross validation, test accuracy was 0.431224 and train accuracy was 0.943098 (test kappa 0.283509, train kappa 0.928658). An SVM was tried as well and did worse on accuracy and F1, so the naive Bayes models were the ones applied to the tweets.

LDA was fit with 7 topics. Average star ratings by topic:

- Topic 3 scored highest at 8.11. The writeup ties this to the human and emotional side of the show.
- Topic 6 scored lowest at 6.73 and was about poor English dubbing.
- Topic 5 (6.80) was about main characters being killed off.

The report concludes that public sentiment was strongly positive, and more positive in the Twitter data than in the IMDb ratings, which supported funding a second season. It also suggested spending more on English dubbing, keeping characters the audience had gotten attached to, and staying with the human side of the story.

## Tools

Python, scikit-learn (MultinomialNB, SVM), NLTK, and sklearn's LatentDirichletAllocation.

## Files

- `code/Final_Project_Alex_Geiger_code.ipynb`
- `reports/Final_Project_Alex_Geiger_final_draft.html`
