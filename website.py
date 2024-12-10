from frontend import app
from train_save_model import train_and_save_model

TRAIN_REQUIRED = False # change to True when running for the first time
DEBUG = True # change to True when debugging

if __name__ == "__main__":
    if TRAIN_REQUIRED:
        # Train and save model
        train_and_save_model(movie_data_path="data/cleaned_data.csv", user_data_path="data/user_data.csv")

    app.run(debug=DEBUG)