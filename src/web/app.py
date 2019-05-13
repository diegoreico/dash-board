import os
import dash
import pandas as pd

from src import settings

print(os.listdir(settings.DATA_FOLDER))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True

ratings_uri = settings.DATA_FOLDER + os.sep + 'ratings.csv'

ratings = pd.read_csv(ratings_uri)

# data for histogram
grouped_ratings = ratings.groupby(['rating']).count()
grouped_ratings['count'] = grouped_ratings.userId
grouped_ratings = grouped_ratings.drop(['userId', 'movieId', 'timestamp'], axis=1)

# movies
movies_uri =  settings.DATA_FOLDER + os.sep + 'movies.csv'
movies = pd.read_csv(movies_uri)
