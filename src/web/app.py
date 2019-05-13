import dash
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True

ratings = pd.read_csv('../data/ratings.csv')

# data for histogram
grouped_ratings = ratings.groupby(['rating']).count()
grouped_ratings['count'] = grouped_ratings.userId
grouped_ratings = grouped_ratings.drop(['userId', 'movieId', 'timestamp'], axis=1)

