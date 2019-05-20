import pandas as pd


def hist_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    grouped_ratings = ratings.groupby(['rating']).count()
    grouped_ratings['count'] = grouped_ratings.userId
    grouped_ratings = grouped_ratings.drop(['userId', 'movieId'], axis=1)
    return grouped_ratings


def top_films_by_avg(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    movie_ratings = movies.set_index('movieId').join(ratings.set_index('movieId'))
    avg_rate = movie_ratings[['title', 'rating']]\
        .groupby('title')\
        .mean()\
        .sort_values(by='rating', ascending=False)

    number_of_rates = movie_ratings[['title', 'rating']]\
        .groupby('title')\
        .size()\
        .sort_values(ascending=False)\
        .rename('count')\
        .to_frame()

    final_df = avg_rate.join(number_of_rates)
    final_df['title'] = final_df.index
    final_df['rating'] = final_df.rating.apply(lambda x: round(x, 2))

    return final_df


def obtain_users(ratings: pd.DataFrame):
    return pd.DataFrame({'user': ratings['userId'].unique()})
