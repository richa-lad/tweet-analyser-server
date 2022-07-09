from fastapi import FastAPI
from pydantic import BaseModel
from data_collection.cred import API_KEY, API_SECRET, ACCESS_TOKEN, TOKEN_SECRET, BEARER_TOKEN
from data_collection.RetrieveTweets import create_dataframe
from preprocessing.utils import *
import twitter
import pickle
from keras_preprocessing.sequence import pad_sequences
from tensorflow import keras
from model.Model import create_model
from numpy import asarray, zeros

# define request body that contains user username
class User(BaseModel):
    name: str

# define the respose model that contains classification of user tweets
class Classified(BaseModel):
    results: list

# start the app
app = FastAPI()

# connect to twitter
api = twitter.Api(
    API_KEY, API_SECRET, ACCESS_TOKEN, TOKEN_SECRET
)

@app.get("/")
async def welcome():
    return "Welcome to the API for 'Which Real Housewife are you?'"


@app.post("/classify", response_model=Classified)
async def classify_user_tweets(user: User):
    # use username to pull user tweets and transform tweets into dataframe of the right format
    columns = ["text", "is_quote_status", "retweet_count", "favorite_count", "favorited", "retweeted", "username"]
    handles = [user.name]
    df = create_dataframe(columns, handles, api)

    # preprocess the data in the same way as training data
    df, target_headers = preprocess(df)
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

    # add more vocab to old tokenizer
    tokenizer.fit_on_texts(X1)

    # turn text to sequence of nums
    X1 = tokenizer.texts_to_sequences(X1)
    X1 = pad_sequences(X1, padding='post', maxlen=max_len_test)

    # get X2 (input for the second model)
    X2 = X[feature_headers[1:]].values

    glove_file = open('glove/glove.twitter.27B.200d.txt', encoding="utf8")
    embeddings_dictionary = dict()
    vocab_size = len(tokenizer.word_index)+1
    print("reading glove file")
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()
    print("creating embedding matrix")
    embedding_matrix = zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    print("creating model from saved checkpoint")
    # create the model from the defined architecture and load the weights
    model = create_model(max_len_test, vocab_size, feature_headers, target_headers, embedding_matrix)
    model.load_weights("checkpoints/training.ckpt")
    print("predicting...")
    # get predictions
    predictions = model.predict(x=[X1, X2])

    # predictions are one hot vectors which need to be transformed back into housewives
    processed_predictions = [target_headers[prediction.index(1)] for prediction in predictions]

    return {"results": processed_predictions}