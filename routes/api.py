import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_collection.RetrieveTweets import create_dataframe
from model.Model import create_model
from preprocessing.utils import preprocess, preprocess_text
from pydantic import BaseModel
import twitter
from utils import return_prediction
import pickle
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from numpy import asarray, zeros
import json
import os

# define request body that contains user username
class User(BaseModel):
    name: str

class Parameters(BaseModel):
    max_len_test: int
    vocab_size: int
    feature_headers: list
    target_headers: list

# start the app
app = FastAPI()

# allow for CORS from front end app
origins = [
    "http://localhost:3000/",
    "http://localhost:3001/",
    "https://rhobh-ta.herokuapp.com",
    "https://tweet-analyzer.onrender.com"
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


@app.post("/setup", response_model=Parameters)
async def setup(user: User):
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

    # turn text to sequence of nums
    X1 = tokenizer.texts_to_sequences(X1)
    X1 = pad_sequences(X1, padding='post', maxlen=max_len_test)

    # get X2 (input for the second model)
    X2 = X[feature_headers[1:]].values
    vocab_size = len(tokenizer.word_index)+1
    
    with open('checkpoints/X1.pickle', 'wb') as handle1:
      pickle.dump(X1, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    with open('checkpoints/X2.pickle', 'wb') as handle2:
      pickle.dump(X2, handle2, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "max_len_test": max_len_test,
        "vocab_size": vocab_size,
        "feature_headers": feature_headers,
        "target_headers": target_headers,
    }



@app.post("/classify")
async def classify(model_parameters: Parameters):

    with open('checkpoints/embeddings.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
    
    with open('checkpoints/X1.pickle', 'rb') as handle1:
        X1 = pickle.load(handle1)

    with open('checkpoints/X2.pickle', 'rb') as handle2:
        X2 = pickle.load(handle2)

    print("creating model")
    # create the model from the defined architecture and load the weights
    model = create_model(
        model_parameters.max_len_test, 
        model_parameters.vocab_size, 
        model_parameters.feature_headers, 
        model_parameters.target_headers, 
        embedding_matrix
    )
    
    print("loading weights")
    model.load_weights("checkpoints/training.ckpt")
    results = return_prediction(
        model, 
        model_parameters.target_headers, 
        X1, 
        X2
    )

    return results
