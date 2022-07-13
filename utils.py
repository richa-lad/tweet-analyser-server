import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import re
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import Concatenate
from keras.models import Model

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