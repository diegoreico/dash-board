import dash

import src.services.DataManipulationService as dataManipulationService
import src.services.RepositoryService as repositoryService

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.config.suppress_callback_exceptions = True

ratings = repositoryService.loadRatings()
# data for histogram
grouped_ratings = dataManipulationService.hist_ratings(ratings)

# movies
movies = repositoryService.loadMovies()
topFilms = dataManipulationService.top_films_by_avg(ratings, movies)

users = dataManipulationService.obtain_users(ratings)

