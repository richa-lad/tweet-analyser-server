from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import Concatenate
from keras.models import Model
from numpy import asarray
from numpy import zeros
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import re
import pickle

def load_data(path):
    return pd.read_csv(path)

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


if __name__ == "__main__":
    df = load_data("data_collection/tweets.csv")
    df, target_headers = preprocess(df)
    feature_headers = ["text", "is_quote_status", "retweet_count", "favorite_count", "favorited", "retweeted"]
    
    X_train, X_test, y_train, y_test = train_test_split(df[feature_headers], df[target_headers], random_state=1)

    X1_train = []
    sentences = list(X_train["text"])
    for sen in sentences:
        X1_train.append(preprocess_text(sen))
    max_len_train = max([len(x) for x in X1_train])

    X1_test = []
    sentences = list(X_test["text"])
    for sen in sentences:
        X1_test.append(preprocess_text(sen))
    max_len_test = max([len(x) for x in X1_test])


    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X1_train)

    X1_train = tokenizer.texts_to_sequences(X1_train)
    X1_test = tokenizer.texts_to_sequences(X1_test)

    vocab_size = len(tokenizer.word_index) + 1

    maxlen = max(max_len_train, max_len_test)
    X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
    X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)


    embeddings_dictionary = dict()

    glove_file = open('glove/glove.twitter.27B.200d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions

    glove_file.close()

    embedding_matrix = zeros((vocab_size, 200))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    
    with open('embeddings.pickle', 'wb') as handle1:
      pickle.dump(embedding_matrix, handle1, protocol=pickle.HIGHEST_PROTOCOL)

    X2_train = X_train[feature_headers[1:]].values
    X2_test = X_test[feature_headers[1:]].values

    model = create_model()
    print(model.summary())

    checkpoint_path = "checkpoints/training.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    history = model.fit(
        x=[X1_train, X2_train],
        y=y_train, batch_size=128, 
        epochs=30, verbose=1, 
        validation_split=0.1,
        callbacks=[cp_callback]
    )
    y_pred = model.predict(x=[X1_test, X2_test])
    score = model.evaluate(x=[X1_test, X2_test], y=y_test, verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])


    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()