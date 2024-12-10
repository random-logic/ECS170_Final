from frontend import app
from train_save_model import *

TRAIN_REQUIRED = False # change to True when running for the first time
DEBUG = True

if __name__ == "__main__":
    if TRAIN_REQUIRED:
        movie_data_path = "data/cleaned_data.csv"
        user_data_path = "data/user_data.csv"

        # Train and save model
        train_and_save_model(movie_data_path, user_data_path)

    app.run(debug=DEBUG)