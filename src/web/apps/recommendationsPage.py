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


global_latent_factors = 10
global_learning_rate = 0.001
global_number_epochs = 10

ratings_matrix = ratings.pivot_table(index=['userId'], columns=['movieId'], fill_value=0.0).values
model = SGD(ratings_matrix, global_latent_factors, np.double(global_learning_rate), global_number_epochs)

layout = html.Div([
    html.H3('Recommendations page'),
    html.A("Go Back", href="/"),
    html.Div([
        html.Div([
            dcc.Markdown('''**Model**: ''',
                         dangerously_allow_html=True),
        ], className="two columns"),
        html.Div([
            html.Button('Train Model', id='button-train')
        ], className="two columns"),
        html.Div([
            dcc.Markdown('Status: `no existe un modelo entrenado`', id='msg-model')
        ], className="three columns"),
        html.Div([
            dcc.Markdown('', id='msg-training')
        ], className="threecolumns"),
    ], className="row"),
    html.Div([
        dcc.Markdown('Number of latent factors',
                     dangerously_allow_html=True),

        dcc.Slider(id='slider-latent', min=0, max=50, marks={i: '{}'.format(i) for i in range(100)}, value=10),
        dcc.Markdown('<br/>Learning rate',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-learning', min=0.0001, max=1, step=0.0333,
                   marks={i: '{:.3f}'.format(i) for i in np.linspace(0.0001, 1, 30, endpoint=False)}, value=0.0001),
        dcc.Markdown('<br/>Number of epochs',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-epochs', min=0, max=50, marks={i: '{}'.format(i) for i in range(100)}, value=10),
    ]),
    html.Div([
        html.Br(),
        html.H4('''AvailableUsers'''),
        html.Div([
            create_table('users-table', users)
        ], className="two columns"),
        html.Div([
            html.Div(id='datatable-interactivity-container')
            # create_hist_genres()
        ], className="nine columns"),
    ], className="row"),
    html.Button('Get Recommendations for users', id='button-predict'),
    dcc.Markdown('', id='msg-predictions'),
    dcc.Interval(id="interval", interval=1 * 1000, n_intervals=0),
    dcc.Markdown('', id='msg-none'),
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
    Output('msg-none', 'children'),
    [Input('button-train', 'n_clicks')],
    [State('slider-latent', 'value'),
    State('slider-learning', 'value'),
    State('slider-epochs', 'value')]
)
def update_model_state_button_click(n_clicks, slider_latent, slider_learning, slider_epochs):
    if n_clicks is None:
        n_clicks = 0

    if n_clicks > 0 and not model.is_training:
        model.train(n_factors=slider_latent,
                    alpha=np.double(slider_learning,dtype=np.double),
                    n_epochs=slider_epochs)
        print('done')


@app.callback(Output('msg-model', 'children'),
              [Input('interval', 'n_intervals')])
def update_metrics(n):
    if model.is_train:
        return 'Exists a trained model'
    else:
        return 'No trained model available'


@app.callback(Output('msg-training', 'children'),
              [Input('interval', 'n_intervals')])
def update_metrics(n):
    if model.is_training:
        return 'Training a new model...'
    else:
        return ''

@app.callback(
    Output('msg-predictions', 'children'),
    [Input('button-predict', 'n_clicks')],
    [State('users-table', "derived_virtual_data"),
     State('users-table', "derived_virtual_selected_rows")]
)
def update_model_state_button_click(n_clicks, rows, derived_virtual_selected_rows):

    if n_clicks is None:
        return ''
    else:
        if model.is_train:
            selected_users = users.iloc[derived_virtual_selected_rows, 0].values
            print(selected_users)
            input = selected_users
            model.predict(input)
            return selected_users
