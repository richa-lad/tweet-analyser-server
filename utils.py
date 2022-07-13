import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import re
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import Concatenate
from keras.models import Model

def get_tweets(api=None, screen_name=None):
    timeline = api.GetUserTimeline(screen_name=screen_name, count=200)
    earliest_tweet = min(timeline, key=lambda x: x.id).id

    while True:
        tweets = api.GetUserTimeline(
            screen_name=screen_name, max_id=earliest_tweet, count=200
        )
        new_earliest = min(tweets, key=lambda x: x.id).id

        if not tweets or new_earliest == earliest_tweet:
            break
        else:
            earliest_tweet = new_earliest
            timeline += tweets

    return timeline

def create_dataframe(columns, handles, api):
    data = []   
    for handle in handles:
        print(f"Saving tweets from account @{handle}")
        timeline = get_tweets(api, handle)
        for tweet in timeline:
            tweet = tweet._json # convert to py dict
            row = list()
            row.append(tweet["text"].replace("\n", "").replace("\t", "").replace("\r", ""))
            row += [tweet[col] for col in columns if col not in ("username", "text")]
            row.append(handle)

            data.append(row)

    df = pd.DataFrame(data=data, columns=columns)

    return df

def preprocess(df):

    df.replace(True, 1, inplace=True)
    df.replace(False, 0, inplace=True)

    lb = LabelBinarizer()
    lb_results = lb.fit_transform(df["username"])
    lb_df = pd.DataFrame(lb_results, columns=lb.classes_)

    result_df = pd.concat([df, lb_df], axis=1)

    return result_df, lb.classes_


def preprocess_text(sen):

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def create_model(maxlen, vocab_size, feature_headers, target_headers, embedding_matrix):

    input_1 = Input(shape=(maxlen,))

    input_2 = Input(shape=(len(feature_headers[1:]),))

    embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], trainable=False)(input_1)
    LSTM_Layer_1 = LSTM(512, return_sequences=True)(embedding_layer)
    LSTM_Layer_2 = LSTM(128, return_sequences=True)(LSTM_Layer_1)
    LSTM_Layer_3 = LSTM(128, return_sequences=True)(LSTM_Layer_2)
    LSTM_Layer_4 = LSTM(128)(LSTM_Layer_3)


    dense_layer_1 = Dense(250, activation='relu')(input_2)
    dense_layer_2 = Dense(125, activation='relu')(dense_layer_1)

    concat_layer = Concatenate()([LSTM_Layer_4, dense_layer_2])
    dense_layer_3 = Dense(125, activation='relu')(concat_layer)
    output = Dense(len(target_headers), activation='softmax')(dense_layer_3)
    model = Model(inputs=[input_1, input_2], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def return_prediction(model, target_headers, X1, X2):
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