# Graduate data science portfolio

These are projects I completed for my M.S. data science coursework at Syracuse University's School of Information Studies. They include a movie inventory database, models of crime risk and high school graduation rates, and two text mining projects.

| Project | Course | Result | Folder |
| --- | --- | --- | --- |
| Movie inventory database | IST 659, Database Management | A member-facing movie database in third normal form, with SQL views and procedures and R plots, built on simulated data. | [IST-659-database-management](IST-659-database-management/) |
| Crime risk modeling | IST 687, Intro to Data Science | San Francisco crime data joined with census demographics, then a support vector model of crime per day by area. | [IST-687-crime-risk-modeling](IST-687-crime-risk-modeling/) |
| High school graduation rates | IST 718, Big Data Analytics | A LightGBM model of census-tract graduation rates had a 5-fold test RMSE of 0.063. Median age and family income were the strongest features. | [IST-718-graduation-rates](IST-718-graduation-rates/) |
| Squid Game sentiment | IST 736, Text Mining | A naive Bayes model trained on 1,186 IMDb reviews and applied to 80,019 tweets. Predicted sentiment was higher on Twitter (average 7.9 vs 7.2). | [IST-736-squid-game-sentiment](IST-736-squid-game-sentiment/) |
| Phrase-level movie review sentiment | IST 664, Natural Language Processing | Naive Bayes on Rotten Tomatoes phrases, with determiner words removed. Best 10-fold average was accuracy 0.630 and F1 0.619. | [IST-664-natural-language-processing](IST-664-natural-language-processing/) |

## Skills

- Languages: Python, R, SQL
- ML/stats: linear regression, random forests, LightGBM, support vector machines, naive Bayes, LDA topic modeling, k-fold cross-validation
- Data/DB: SQL Server, Microsoft Access, pandas, dplyr
- Tools: Jupyter, scikit-learn, NLTK, statsmodels, ggplot2, seaborn, leaflet

The original data files are not in the repo, so the notebooks and R scripts need those sources before they will run through. The license is MIT, carried over from the original IST 659 project.
