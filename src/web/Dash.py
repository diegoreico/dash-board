import dash

from src.web.LayoutService import LayoutService

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
name = "Recommendation System Dashboard App"

app = dash.Dash(name, external_stylesheets=external_stylesheets)
app.layout = LayoutService(app).obtainLayout()

server = app.server

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)