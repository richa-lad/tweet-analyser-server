from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import twitter
from utils import create_dataframe, preprocess, preprocess_text, create_model
import pickle
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from numpy import asarray, zeros
import json
import os

# define request body that contains user username
class User(BaseModel):
    name: str

# start the app
app = FastAPI()

# allow for CORS from front end app
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_SECRET"]
ACCESS_TOKEN = os.environ["ACCESS_TOKEN"]
TOKEN_SECRET = os.environ["TOKEN_SECRET"]

# connect to twitter
api = twitter.Api(
    API_KEY, API_SECRET, ACCESS_TOKEN, TOKEN_SECRET
)

@app.get("/")
async def welcome():
    return "Welcome to the API for 'Which Real Housewife are you?'"


@app.post("/classify")
async def classify_user_tweets(user: User):
    # use username to pull user tweets and transform tweets into dataframe of the right format
    columns = ["text", "is_quote_status", "retweet_count", "favorite_count", "favorited", "retweeted", "username"]
    handles = [user.name]
    df = create_dataframe(columns, handles, api)

    # preprocess the data in the same way as training data
    df, _ = preprocess(df)
    target_headers = ['DENISE_RICHARDS', 'GarcelleB', 'KyleRichards', 'SuttonBStracke', 'YolandaHadid', 'crystalsminkoff',
        'doritkemsley1', 'erikajayne', 'lisarinna']
    feature_headers = columns[:-1]
    X = df[feature_headers]

    # get X1 (input for the first model)
    X1 = []
    sentences = list(X["text"])
    for sen in sentences:
        X1.append(preprocess_text(sen))
    max_len_test = max([len(x) for x in X1])

    # loading the tokenizer created in training
    with open('checkpoints/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('checkpoints/embeddings.pickle', 'rb') as handle1:
        embedding_matrix = pickle.load(handle1)

    # turn text to sequence of nums
    X1 = tokenizer.texts_to_sequences(X1)
    X1 = pad_sequences(X1, padding='post', maxlen=max_len_test)

    # get X2 (input for the second model)
    X2 = X[feature_headers[1:]].values

    vocab_size = len(tokenizer.word_index)+1
    
    print("creating model")
    # create the model from the defined architecture and load the weights
    model = create_model(max_len_test, vocab_size, feature_headers, target_headers, embedding_matrix)
    print("loading weights")
    model.load_weights("checkpoints/training.ckpt")
    print("predicting...")

    # get predictions
    predictions = model.predict(x=[X1, X2])
    one_hot = [[1 if max(prediction)==x else 0 for x in prediction] for prediction in predictions]
    processed_to_names = [target_headers[vec.index(1)] for vec in one_hot]

    scores_names = {processed_to_names.count(name)/len(processed_to_names):name for name in processed_to_names}

    # return results in easy to consume format for js
    scores = list(scores_names.keys())
    scores.sort(reverse=True)
    results = dict()
    for i, score in enumerate(scores[:5]):
        results[f"{i}"] = {
            "username": scores_names[score],
            "score": score
        }

    return results