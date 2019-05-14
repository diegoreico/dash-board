import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np

from dash.dependencies import Input, State
from dash.dependencies import Output

from src.services.SGD import SGD
from src.web.app import app, users, ratings, grouped_ratings, movies


def create_table(component_id, data: pd.DataFrame, selectable: str = "multi"):
    return dash_table.DataTable(
        id=component_id,
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict("rows"),
        row_selectable=selectable,
        sorting=True,
        filtering=True,
        pagination_mode="fe",
        pagination_settings={
            "displayed_pages": 1,
            "current_page": 0,
            "page_size": 10,
        },
        navigation="page",
    )


def create_hist_ratings(dff):
    return dcc.Graph(
        figure=go.Figure(
            data=[
                go.Bar(
                    x=dff.index,
                    y=dff['count'],
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


model = SGD(ratings, n_factors=30, alpha=np.double(0.01), n_epochs=10)

layout = html.Div([
    html.H3('Recommendations page'),
    html.A("Go Back", href="/"),
    html.Div([
        html.Div([
            dcc.Markdown('''Model: ''',
                         dangerously_allow_html=True),
        ], className="two columns"),
        html.Div([
            html.Button('Train Model', id='button')
        ], className="one columns"),
        html.Div([
            dcc.Markdown('Status: `no model trained`', id='msg-training')
        ], className="two columns"),
    ], className="row"),
    html.Div([
        dcc.Markdown('''AvailableUsers''',
                     dangerously_allow_html=True),
        html.Div([
            create_table('users-table', users)
        ], className="two columns"),
        html.Div([
            html.Div(id='datatable-interactivity-container')
            # create_hist_genres()
        ], className="nine columns"),
    ], className="row"),
    dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0)
])


@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('users-table', "derived_virtual_data"),
     Input('users-table', "derived_virtual_selected_rows")])
def update_graph(rows, derived_virtual_selected_rows):
    if derived_virtual_selected_rows is None:
        return html.Div([])
    else:
        selected_users = users.iloc[derived_virtual_selected_rows, 0].values

        filteredFilmsRatings = obtain_film_rates_for_users(selected_users)

        dff = filteredFilmsRatings[['userId', 'title', 'rating']]

        return html.Div(
            [create_table('selected-rates-tables', dff, None)]
        )


def obtain_film_rates_for_users(selected_users):
    filter = ratings['userId'].isin(selected_users)
    filteredRatings = ratings[filter]
    filteredFilmsRatings = filteredRatings.set_index('movieId') \
        .join(movies.set_index('movieId'))
    return filteredFilmsRatings


@app.callback(
    Output('msg-training', 'children'),
    [Input('button', 'n_clicks')])
def update_model_state_button_click(n_clicks):
    if n_clicks is None:
        n_clicks = 0

    if n_clicks > 0 and not model.is_training:
        model.train()
        print("finish")
