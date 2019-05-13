import os
from dotenv import load_dotenv

# OR, the same with increased verbosity:
load_dotenv(verbose=True)

DATA_FOLDER = os.getenv("DATA_FOLDER", os.path.join("data"))