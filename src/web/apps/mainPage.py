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
    html.Div(id='datatable-interactivity-container'),
    html.Div(id='app-1-display-value'),
    dcc.Link('Go to Index', href='/')
])

@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('main-table', "derived_virtual_data"),
     Input('main-table', "derived_virtual_selected_rows")])
def update_graph(rows, derived_virtual_selected_rows):
    # When the table is first rendered, `derived_virtual_data` and
    # `derived_virtual_selected_rows` will be `None`. This is due to an
    # idiosyncracy in Dash (unsupplied properties are always None and Dash
    # calls the dependent callbacks when the component is first rendered).
    # So, if `rows` is `None`, then the component was just rendered
    # and its value will be the same as the component's dataframe.
    # Instead of setting `None` in here, you could also set
    # `derived_virtual_data=df.to_rows('dict')` when you initialize
    # the component.
    if derived_virtual_selected_rows is None:
        derived_virtual_selected_rows = []

    if rows is None:
        dff = ratings
    else:
        dff = pd.DataFrame(rows)

    colors = []
    for i in range(len(dff)):
        if i in derived_virtual_selected_rows:
            colors.append("#7FDBFF")
        else:
            colors.append("#0074D9")

    return html.Div(
        [
            dcc.Graph(
                id=column,
                figure={
                    "data": [
                        {
                            "x": dff["rating"],
                            # check if column exists - user may have deleted it
                            # If `column.deletable=False`, then you don't
                            # need to do this check.
                            "y": dff[column]
                            "type": "bar",
                            "marker": {"color": colors},
                        }
                    ],
                    "layout": {
                        "xaxis": {"automargin": True},
                        "yaxis": {"automargin": True},
                        "height": 250,
                        "margin": {"t": 10, "l": 10, "r": 10},
                    },
                },
            )
            for column in ["pop", "lifeExp", "gdpPercap"]
        ]
    )

