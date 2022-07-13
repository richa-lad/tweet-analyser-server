from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import twitter
from utils import create_dataframe, preprocess, preprocess_text, create_model, return_prediction
import pickle
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from numpy import asarray, zeros
import json
import os
from rq import Queue
from worker import conn

# add function to background queue using redis
q = Queue(connection=conn)
result = q.enqueue(return_prediction, 'http://heroku.com')

# define request body that contains user username
class User(BaseModel):
    name: str

# start the app
app = FastAPI()

# allow for CORS from front end app
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://rhobh-ta.herokuapp.com"
]

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
    
    results = return_prediction(model, target_headers, X1, X2)

    return results