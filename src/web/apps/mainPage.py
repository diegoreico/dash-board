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
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to Index', href='/')
])


@app.callback(
    Output('app-1-display-value', 'children'),
    [Input('app-1-dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)
