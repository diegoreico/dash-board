# Recommendation System Dashboard

**Explain that this application uses movielns dataset to perform group recommendations**

## How to run this application

1. Install all application requirements using pip
```python
pip install -r requirements.txt
```
2. Get into the src folder and run the app using python
```bash
cd src
python3 app.py
```

After starting the app, a web dashboard should be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/). That web page shows basic data insights on the root page and if you go to the url [http://127.0.0.1:8050/recommendations](http://127.0.0.1:8050/recommendations), you can train a recommendation model selecting his hyper-parameters.

Once the model is trained, you can select a group of users from the the table that is located under the model hyper-parameters. Another table will show selected users performed ratings. Once you are done creating a group, you can click on the button `GET RECOMMENDATIONS FOR USERS` to generate group recommendations, which will be loaded into another table at the end of the page

## Data Used  

Data used in this application is located under the folder `data`, that contains the following files:

- `movies.csv`: contains movie id, title and genre for all available movies
- `ratings.csv`: contains the user id, timestamp and rating performed by each user to a movie 

## Matrix Factorization Algorithm

Matrix factorization is performed using a self made implementation of the SVD++ algorithm, wich is used to perform matrix factorization on the ratings matrix.

## How group recommendations are performed

Group recommendation are obtained `Least Misery Criteria` over the individual recommendation of each user inside a group