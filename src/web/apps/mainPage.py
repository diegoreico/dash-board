import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd

from dash.dependencies import Input, Output

from src.web.app import app
from src.web.app import ratings


layout = html.Div([
    html.H3('App 1'),
    dash_table.DataTable(
        id = 'main-table',
        columns = [{"name": i, "id": i} for i in ratings.columns],
        data = ratings.to_dict("rows"),
        filtering=True,
        row_selectable="multi",
        pagination_mode="fe",
        pagination_settings={
            "displayed_pages": 1,
            "current_page": 0,
            "page_size": 20,
        },
        navigation="page",
    ),
    html.H5('Histograma de las valoraciones de los usuarios'),
    # dcc.Graph(
    #     id='hist',
    #     figure={
    #         "data": [
    #             {
    #                 "x": ratings["rating"],
    #                 # check if column exists - user may have deleted it
    #                 # If `column.deletable=False`, then you don't
    #                 # need to do this check.
    #                 "y": ratings["rating"],
    #                 "type": "bar",
    #             }
    #         ],
    #         "layout": {
    #             "xaxis": {"automargin": True},
    #             "yaxis": {"automargin": True},
    #             "height": 250,
    #             "margin": {"t": 10, "l": 10, "r": 10},
    #         },
    #     },
    # ),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to Index', href='/')
])



