# Predicting high school graduation rates

IST 718, Big Data Analytics. September 2022. Professor Lando.

Group 8: Brett Bastianelli, Alex Geiger, Maria Ng, and Samantha White (the notebook lists her as Sam White).

We used census and related data to model high school graduation rates and to see which local conditions moved with them. The report cites an NCES figure of 88% for the U.S. rate in the 2021-2022 school year. We were interested in the zip codes and tracts that sit well below that.

The paper is [reports/Group_8_Final_Project_doc.pdf](reports/Group_8_Final_Project_doc.pdf). The slides and a knitted copy of the notebook are in `reports/` as well.

## Data

Census files were pulled from www2.census.gov (gazetteer files, census tables, and column-name files) and combined with property tax and lottery revenue. After cleaning, each row was a U.S. zip code. NAs were set to 0, and duplicates were collapsed by taking the median within zip code. For the later LightGBM work we switched to census tracts, which are smaller. The notebook notes about 3.8 tracts per zip code on average.

## Models

- OLS on zip codes (divorced, median age, family median, population, state, not in the labor force, elderly), with the graduation rate squared: adjusted R-squared of 0.623.
- The same idea grouped by state (divorce, median age, not in the labor force, unemployed, property tax): adjusted R-squared of 0.746. Property tax was not significant (p-value 0.196). Lottery and property tax also did not have a significant effect in the earlier linear models.
- Random forest: RMSE of 0.00047, as reported in the paper.
- LightGBM with 5-fold cross validation, trained on tracts. Average test scores from the notebook: MAE 0.044433, MAPE 0.058086, RMSE 0.062614.

Median age was the strongest feature in the LightGBM models, followed by median family income in the feature plot (the written summary also highlights the share of people not in the labor force). Both age and family income were positively related to graduation rates.

We also used the models to look at a California census tract with a 28% graduation rate, then asked what would happen if family income or age shifted. The writeup recommends tax incentives for higher-paying work in the area and government-sponsored job fairs.

## Tools

Python, pandas, scikit-learn, LightGBM, and seaborn. The notebook is [`code/IST_718_Final_Project.ipynb`](code/IST_718_Final_Project.ipynb).

## Files

- `code/IST_718_Final_Project.ipynb`
- `reports/Group_8_Final_Project_doc.pdf`
- `reports/Group_8_Final_Project_ppt.pdf`
- `reports/Group_8_Final_Project_notebook.html`
