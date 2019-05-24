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


def create_errors_plot():
    return dcc.Graph(
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(model.epoch_errors))),
                    y=model.epoch_errors,
                    mode='lines+markers',
                    name='Model Epoch Errors',
                ),
                go.Scatter(
                    x=list(range(len(model.epoch_test_errors))),
                    y=model.epoch_test_errors,
                    mode='lines+markers',
                    name='Model Epoch Test Errors',
                )
            ],
            layout=go.Layout(
                title='Training errors',
                showlegend=True,
                legend=go.layout.Legend(
                    x=0,
                    y=1.0
                ),
                margin=go.layout.Margin(l=40, r=0, t=40, b=30)
            )
        ),
        style={'height': 300},
        id='error-line-plot'
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
        ], className="two columns"),
        html.Div([
            dcc.Markdown('', id='msg-epochs')
        ], className="two columns"),
        html.Div([], id='plot-errors', className="twelve columns"),
    ], className="row"),
    dcc.Markdown('''**Model Hyperparameters**: '''),
    html.Div([
        dcc.Markdown('Number of latent factors',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-latent', min=5, max=100, step=5,
                   marks={i: '{}'.format(i) for i in range(5, 100, 5)}, value=10),
        dcc.Markdown('<br/>Learning rate',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-learning', min=0, max=0.1, step=0.05, value=0.05,
                   marks={i: '{:.4f}'.format(i) for i in np.linspace(0, 0.1, 20, endpoint=False)}),
        dcc.Markdown('<br/>Bias Regularization',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-bias', min=0, max=1, step=0.05, value=0.05,
                   marks={i: '{:.2f}'.format(i) for i in np.linspace(0, 1, 20, endpoint=False)}),
        dcc.Markdown('<br/>L2 Regularization',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-l2', min=0, max=1, step=0.05, value=0.05,
                   marks={i: '{:.2f}'.format(i) for i in np.linspace(0, 1, 20, endpoint=False)}),
        dcc.Markdown('<br/>Number of epochs',
                     dangerously_allow_html=True),
        dcc.Slider(id='slider-epochs', min=0, max=1000, step=20,
                   marks={i: '{}'.format(i) for i in range(0, 1000, 20)}, value=20),
    ]),
    html.Div([
        html.Br(),
        html.H4('''AvailableUsers'''),
        html.Div([
            create_table('users-table', users)
        ], className="three columns"),
        html.Div([
            html.Div(id='datatable-interactivity-container')
        ], className="seven columns"),
    ], className="row"),
    html.H4('Predicted Ratings'),
    html.Button('Get Recommendations for users', id='button-predict'),
    html.Div([], id='msg-predictions'),
    dcc.Interval(id='interval', interval=5 * 1000, n_intervals=0),
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
     State('slider-epochs', 'value'),
     State('slider-bias', 'value'),
     State('slider-l2', 'value')]
)
def update_model_state_button_click(n_clicks, slider_latent, slider_learning, slider_epochs, slider_bias, slider_l2):
    if n_clicks is None:
        n_clicks = 0

    if n_clicks > 0 and not model.is_training:
        model.train(n_factors=slider_latent,
                    learning_rate=np.double(slider_learning, dtype=np.double),
                    n_epochs=slider_epochs,
                    bias_reg=slider_bias,
                    l2_reg=slider_l2)
        print('done')


@app.callback(Output('msg-model', 'children'),
              [Input('button-train', 'n_clicks'),
               Input('interval', 'n_intervals')])
def update_model_exists_msg(clicks, intervals):
    if model.is_train:
        return 'Exists a trained model'
    else:
        return 'No trained model available'


@app.callback(Output('msg-training', 'children'),
              [Input('button-train', 'n_clicks'),
               Input('interval', 'n_intervals')])
def update_model_training_msg(clicks, intervals):
    if model.is_training:
        return 'Training a new model...'
    else:
        return ''


@app.callback(Output('msg-epochs', 'children'),
              [Input('button-train', 'n_clicks'),
               Input('interval', 'n_intervals')])
def update_epochs_msg(clicks, intervals):
    if model.is_training:
        return 'Current epoch: {}'.format(model.current_epoch)
    else:
        return ''


@app.callback(
    Output('msg-predictions', 'children'),
    [Input('button-predict', 'n_clicks')],
    [State('users-table', "derived_virtual_data"),
     State('users-table', "derived_virtual_selected_rows")]
)
def update_table_with_recommendations_from_selected_users(n_clicks, rows, derived_virtual_selected_rows):
    if n_clicks is None:
        return ''
    else:
        if model.is_train:
            selected_users = users.iloc[derived_virtual_selected_rows, 0].values
            input = selected_users
            group_recommendations, group_recommendations_indexes = model.obtain_group_recommendations(input)

            recommendation_results = movies.loc[group_recommendations_indexes]
            recommendation_results['predicted_rating'] = group_recommendations[group_recommendations_indexes]

            return html.Div([
                create_table('table-group-recommendations', recommendation_results)
            ])


@app.callback(
    Output('plot-errors', 'children'),
    [Input('interval', 'n_intervals')],
)
def update_graph_over_time(interval):
    if len(model.epoch_errors) > 0:
        return html.Div([
            create_errors_plot()
        ])
    else:
        return html.Div([])
