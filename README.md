# Graduate data science coursework

These are projects I completed for graduate data science coursework at Syracuse University's School of Information Studies. The earliest commits here use my syr.edu email, from 2020.

I started this repository from my IST 659 project so those commits are still the root of the history. The other course projects were merged in with their original commits intact.

| Course | Project | What it is | Tools | Folder |
| --- | --- | --- | --- | --- |
| IST 659, Data Administration Concepts and Database Management | Movie inventory database | Member-facing movie database with SQL, Access forms, and R plots | SQL Server, Access, R | [IST-659-database-management](IST-659-database-management/) |
| IST 687, Introduction to Data Science | Crime risk modeling | Exploratory analysis and a model of crime risk in San Francisco, using demographics and location | R | [IST-687-crime-risk-modeling](IST-687-crime-risk-modeling/) |
| IST 718, Big Data Analytics | High school graduation rates | Models of graduation rates from census and related data, with a short policy writeup | Python, LightGBM, scikit-learn | [IST-718-graduation-rates](IST-718-graduation-rates/) |
| IST 736, Text Mining | Squid Game sentiment | Sentiment models on IMDb reviews and tweets, plus topic modeling | Python, scikit-learn, NLTK | [IST-736-squid-game-sentiment](IST-736-squid-game-sentiment/) |
| NLP course (final project) | Phrase-level movie review sentiment | Naive Bayes on Rotten Tomatoes phrases, with a determiner-removal feature | Python, scikit-learn, NLTK | [NLP-final-project](NLP-final-project/) |

Work across the courses includes SQL and relational design, R, Python, regression and tree models, text classification, and topic modeling.

## Layout

```
graduate-data-science-portfolio/
├── IST-659-database-management/
├── IST-687-crime-risk-modeling/
├── IST-718-graduation-rates/
├── IST-736-squid-game-sentiment/
├── NLP-final-project/
├── LICENSE
└── README.md
```

Each project folder has `code/` and `reports/`. The 659 project also has `docs/` and `images/` for the logical model.

## Running the code

The projects were written at different times, so the setups are not the same, and most of the data files were never checked into the original repos.

- IST 659: SQL Server for the scripts under `code/`. Run the design script, then the simulated data, then the engineering script (or use `code/main.sql`, which combines them). The R plots in `code/analysis/` expect an ODBC connection named `dbase64`.
- IST 687: R or RStudio. Open `code/Project_Summary_1.Rmd`. It still points at a local CSV path from my machine at the time.
- IST 718 and the two text projects: Python and Jupyter. You will need the original data sources before the notebooks will run all the way through.

The license at the root is the MIT license from the original 659 repo.
