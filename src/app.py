
import dash

from src.LayoutService import LayoutService

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

name = "Recommendation System Dashboard App"
app = dash.Dash(name, external_stylesheets=external_stylesheets)

server = app.server

app.layout = LayoutService.obtainLayout()

@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)
