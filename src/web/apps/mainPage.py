import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go

from src.web.app import grouped_ratings
from src.web.app import movies
from src.web.app import topFilms



def create_hist_ratings():
    return dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=grouped_ratings.index,
                    y=grouped_ratings['count'],
                    name='valoraciones de los usuarios',
                )
            ],
            layout=go.Layout(
                title='Histograma valoraciones usuarios',
                showlegend=True,
                legend=go.layout.Legend(
                    x=0,
                    y=1.0
                ),
                margin=go.layout.Margin(l=40, r=0, t=40, b=30)
            )
        ),
        style={'height': 300},
        id='hist-ratings'
    )


def create_hist_genres():
    data = movies['genres'].value_counts().sort_values()[-20:]

    return dcc.Graph(
        figure=go.Figure(
            data=[
                go.Pie(
                    labels=data.index,
                    values=data.values,
                    name='películas por género',
                )
            ],
            layout=go.Layout(
                title='Top 20 Géneros más populares',
                showlegend=True,
                legend=go.layout.Legend(
                    x=0,
                    y=1.0
                ),
                margin=go.layout.Margin(l=40, r=0, t=40, b=100)
            )
        ),
        style={'height': 300},
        id='hist-genres'
    )


def create_data_table():
    return dash_table.DataTable(
        id='main-table',
        columns=[{"name": i, "id": i} for i in topFilms.columns],
        data=topFilms.to_dict("rows"),
        row_selectable="multi",
        sorting=True,
        filtering=True,
        pagination_mode="fe",
        pagination_settings={
            "displayed_pages": 1,
            "current_page": 0,
            "page_size": 20,
        },
        navigation="page",
    )


layout = html.Div([
    html.H3('Database content'),
    html.A("Go to Recommendations Page", href="/recommendations"),
    html.Div([
        dcc.Markdown('''To filter table values, you can use expressions like `> num(30)` <br/>
        Ex: to get films with more than 250 ratings, use the following expression on the rating column `> num(250)`''',
                     dangerously_allow_html=True),
        html.Div([
            create_data_table()
        ], className="four columns"),
        html.Div([
            create_hist_ratings(),
            create_hist_genres()
        ], className="six columns"),
    ], className="row"),
])
