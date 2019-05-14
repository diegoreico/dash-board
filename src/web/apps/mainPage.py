import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go

import pandas as pd

from dash.dependencies import Input, Output

from src.web.app import app
from src.web.app import ratings
from src.web.app import grouped_ratings
from src.web.app import movies

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

def craete_hist_genres():
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
        columns=[{"name": i, "id": i} for i in ratings.columns],
        data=ratings.to_dict("rows"),
        row_selectable="multi",
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
    html.Div([
        html.Div([
            create_data_table()
        ], className="three columns"),
        html.Div([
            create_hist_ratings(),
            craete_hist_genres()
        ], className="nine columns"),
    ], className="row"),

    html.Div(id='app-1-display-value'),
    dcc.Link('Go to Index', href='/')
])
