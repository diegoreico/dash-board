from src.web.Index import app
from src.web.app import server

import pandas as pd

if __name__ == '__main__':
    app.run_server(debug=True)
