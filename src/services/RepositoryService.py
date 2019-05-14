import os

import pandas as pd

from src import settings


def loadRatings() -> pd.DataFrame:
    ratings_uri = settings.DATA_FOLDER + os.sep + 'ratings.csv'
    return pd.read_csv(ratings_uri)


def loadMovies() -> pd.DataFrame:
    movies_uri = settings.DATA_FOLDER + os.sep + 'movies.csv'
    return pd.read_csv(movies_uri)