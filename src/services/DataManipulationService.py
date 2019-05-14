import pandas as pd


def hist_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    grouped_ratings = ratings.groupby(['rating']).count()
    grouped_ratings['count'] = grouped_ratings.userId
    grouped_ratings = grouped_ratings.drop(['userId', 'movieId', 'timestamp'], axis=1)
    return grouped_ratings
