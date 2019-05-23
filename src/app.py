from dotenv import load_dotenv

env_path = '.env'
load_dotenv(dotenv_path=env_path, verbose=True)

from src.web.Index import app
from src.web.app import server

if __name__ == '__main__':
    app.run_server(debug=False)
