# Graduate data science portfolio

These are projects I completed for my M.S. data science coursework at Syracuse University, plus my capstone for the MIT Professional Education Applied AI and Data Science Program. They include a loan default model, a retail sales forecast, a movie inventory database, models of crime risk and high school graduation rates, and two text mining projects.

| Project | Course | Result | Team or solo | Folder |
| --- | --- | --- | --- | --- |
| Loan default prediction | MIT Professional Education, Applied AI and Data Science Program capstone | A LightGBM model of home equity loan defaults (test AUC 0.971) with a 20% approval cutoff. Estimated net revenue rises from $10.17M to $18.30M while approving 77.7% of applicants. | Solo | [MIT-capstone-loan-default-prediction](MIT-capstone-loan-default-prediction/) |
| Retail sales forecasting | IST 707, Data Analytics | Daily LightGBM forecasts for 30,490 Walmart item and store series (M5 data), 28 days ahead. In a three-window rolling-origin backtest the average WRMSSE was 0.6375, against 0.8938 for a seasonal naive forecast. | Solo | [IST-707-retail-sales-forecasting](IST-707-retail-sales-forecasting/) |
| Movie inventory database | IST 659, Database Management | A member-facing movie database in third normal form, with SQL views and procedures and R plots, built on simulated data. | Solo | [IST-659-database-management](IST-659-database-management/) |
| Crime risk modeling | IST 687, Intro to Data Science | San Francisco crime data joined with census demographics, then a support vector model of crime per day by area. | Group | [IST-687-crime-risk-modeling](IST-687-crime-risk-modeling/) |
| High school graduation rates | IST 718, Big Data Analytics | A LightGBM model of census-tract graduation rates had a 5-fold test RMSE of 0.063. Median age and family income were the strongest features. | Group | [IST-718-graduation-rates](IST-718-graduation-rates/) |
| Squid Game sentiment | IST 736, Text Mining | A naive Bayes model trained on 1,186 IMDb reviews and applied to 80,019 tweets. Predicted sentiment was higher on Twitter (average 7.9 vs 7.2). | Group | [IST-736-squid-game-sentiment](IST-736-squid-game-sentiment/) |
| Phrase-level movie review sentiment | IST 664, Natural Language Processing | Naive Bayes on Rotten Tomatoes phrases, with determiner words removed. Best 10-fold average was accuracy 0.630 and F1 0.619. | Solo | [IST-664-natural-language-processing](IST-664-natural-language-processing/) |

## Skills

- Languages: Python, R, SQL
- ML/stats: time series forecasting, rolling-origin backtesting, linear regression, random forests, LightGBM, support vector machines, naive Bayes, k-means and hierarchical clustering, LDA topic modeling, k-fold cross-validation
- Data/DB: SQL Server, Microsoft Access, pandas, dplyr
- Tools: Jupyter, scikit-learn, NLTK, statsmodels, ggplot2, seaborn, leaflet

The original data files are not in the repo, so the notebooks and R scripts need those sources before they will run through. The license is MIT, carried over from the original IST 659 project.
