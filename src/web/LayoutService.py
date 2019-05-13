
import dash

import dash_core_components as dcc
import dash_html_components as html

class LayoutService:
    def __init__(self, app) -> None:
        super().__init__()
        LayoutService.app = app

    @classmethod
    def obtainLayout(cls) -> html.Div:
        layout = html.Div([
            html.H2('Hello World'),
            dcc.Dropdown(
                id='dropdown',
                options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
                value='LA'
            ),
            html.Div(id='display-value')
        ])

        return layout

