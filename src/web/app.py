import dash

from src.services.DataManipulationService import hist_ratings
from src.services.RepositoryService import loadRatings, loadMovies

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True

ratings = loadRatings()
# data for histogram
grouped_ratings = hist_ratings(ratings)

# movies
movies = loadMovies()

