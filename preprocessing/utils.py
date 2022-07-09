import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import re

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