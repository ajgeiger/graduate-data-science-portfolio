# Movie inventory database

A member-facing movie inventory database, in the same general vein as IMDb, with SQL and R analysis on top of simulated data.

## Course

IST 659, Data Administration Concepts and Database Management. The course covers how to design, build, and manage databases, including data modeling, normalization, and writing SQL for retrieval and performance.

## Problem

I wanted a database members could use to rate movies and record what they watched, and then a way to ask questions about production companies, genre, and ratings by gender.

## Data

The records are simulated. I loaded them with SQL insert scripts, then used views, procedures, and functions to answer the data questions.

## Approach

The schema is in third normal form, with integer identity primary keys. After loading the data I added views, procedures, and functions, and put a Microsoft Access form in front of it for editing records. The R scripts pull from the database over ODBC and plot ratings by genre, movie, company, and gender.

![Logical diagram](images/logical_diagram.png)

## Results

These numbers are from the simulated data, not from real movies.

- The actors with the most comedy credits were tied at two movies each: Chris Navan, Egg Milk, Uni Eldino, and Zach Dandino.
- *Or Did He* had the most female views (eight in the ratings-by-movie chart).
- *The Man Who Knew SQL* had a five-star rating.
- High Water was the highest-rated production company in the SQL questions, with a rating of 3.
- In the genre chart, action came out on top. Female members both rated more often and gave higher ratings.

The writeup is in [reports/final-report.pdf](reports/final-report.pdf). The cover page says IST 658, but the course and the SQL headers are IST 659.

## Tools

SQL Server, Microsoft Access, and R (RODBC, dplyr, ggplot2).

## Files

- `code/main.sql`: full script (tables, data, and later objects)
- `code/design/physical-database-design.sql`: tables
- `code/simulation/simulated-data.sql`: inserts
- `code/engineering/data-engineering.sql`: views, functions, procedures, and the data questions
- `code/analysis/`: R plots
- `docs/logical_model.xml`: logical model
- `images/logical_diagram.png`
- `reports/final-report.pdf`

To rebuild the database, run the design script, then the simulated data, then the engineering script, or use `code/main.sql`, which combines them. The R scripts query an ODBC source named `dbase64`.
