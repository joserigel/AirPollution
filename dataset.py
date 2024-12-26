from dotenv import load_dotenv
import numpy as np
import math
import pandas as pd
import os

load_dotenv()

PATH_TO_DATASET = os.getenv("PATH_TO_DATASET")

def classification_to_binary(classification):
    if classification == "Good":
        return np.array([1, 0, 0, 0])
    elif classification == "Moderate":
        return np.array([0, 1, 0, 0])
    elif classification == "Poor":
        return np.array([0, 0, 1, 0])
    else:
        return np.array([0, 0, 0, 1])


def dataframe_to_numpy(df):
    features = df[df.columns[:-1]]
    label = df["Air Quality"].map(classification_to_binary)
    return features.to_numpy(), np.stack(label.to_numpy()) 

def normalize(dataset, maximum = None, minimum = None):
    if maximum is None:
        maximum = np.max(dataset, axis=0)
    if minimum is None:
        minimum = np.min(dataset, axis=0)
    return (
        (dataset - minimum) / (maximum - minimum),
        maximum, minimum
    )


def split_to_batches(dataset, batch_size):
    features, labels = dataset
    features = np.array_split(features, batch_size)
    labels = np.array_split(labels, batch_size)
    return list(zip(features, labels))
    

def dataset(train_split=0.7, randomized=True):
    df = pd.read_csv(PATH_TO_DATASET, sep=",")
    
    if randomized:
        df = df.sample(frac=1)
        
    rows = df.shape[0]
    train_rows = math.floor(rows * train_split)
    train = df[:train_rows]
    test = df[train_rows:]
    
    train_features, train_labels = dataframe_to_numpy(train)
    test_features, test_labels = dataframe_to_numpy(test)
    return (
        (train_features, train_labels),
        (test_features, test_labels)
    )

if __name__ == "__main__":
    dataset()