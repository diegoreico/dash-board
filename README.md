# Recommendation System Dashboard

This application creates a web dashboard that could be used to generate movie recommendations for groups. It also allows you to tune your own recommendation model selecting it's hyperparameters.

## Requirements

This application uses **python 3**, **pip** and you need to install the dependencies inside file **requiremetns.txt**

## How to run this application

1. Install all application requirements using pip(you need to have pip installed in your computer)
    ```python
    pip install -r requirements.txt
    ```
2. Get into the src folder and run the app using python
    ```bash
    python3 -m src.app 
    ```

After starting the app, a web dashboard should be available at [http://127.0.0.1:8050/](http://127.0.0.1:8050/). That web page shows basic data insights on the root page and if you go to the url [http://127.0.0.1:8050/recommendations](http://127.0.0.1:8050/recommendations), you can train a recommendation model selecting his hyper-parameters.

Once the model is trained, you can select a group of users from the the table that is located under the model hyper-parameters. Another table will show selected users performed ratings. Once you are done creating a group, you can click on the button `GET RECOMMENDATIONS FOR USERS` to generate group recommendations, which will be loaded into another table at the end of the page

## Data Used  

This aplication uses movielens dataset and it is located under the folder `data`, that contains the following files:

- `movies.csv`: contains 193609 different movies with the feilds movie id, title and genre for all available movies
- `ratings.csv`: contains 100836 ratings performed by users and each rating contains the user id, timestamp and rating performed by each user to a movie 

## Matrix Factorization Algorithm

Matrix factorization is performed using a self made implementation of the SVD++ algorithm, wich is used to perform matrix factorization on the ratings matrix.
Implementation can be found on the `train` method of the file [src/services/SGD.py](./src/services/SGD.py)

## How group recommendations are performed

Group recommendation are obtained using the method/criteria `Least Misery Criteria` over the individual recommendation of each user inside a group. Impementation is available on the method `obtain_group_recommendations` inside the file [src/services/SGD.py](./src/services/SGD.py)

## More Details

More details about the recommendation process and used data can be found inside [this google colaboratory notebook](https://colab.research.google.com/drive/1BI5xxtrF0NMVhexUbUP5smVHb9h3oiAa). You can read an offline rendered version of that notebook on one of the following files:
- [SistemaDeRecomendaciónParaGrupos.ipynb](SistemaDeRecomendaciónParaGrupos.ipynb)
- [SistemaDeRecomendaciónParaGrupos.pdf](SistemaDeRecomendaciónParaGrupos.pdf)
