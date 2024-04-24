#import all the required packages/libraries
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import requests
from sklearn.linear_model import LogisticRegression
from sklearn import utils
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

CLASSES = ['PREAMBLE', 'NONE', 'FAC', 'ARG_RESPONDENT', 'RLC', 'ARG_PETITIONER', 'ANALYSIS', 'PRE_RELIED', 'RATIO', 'RPC', 'ISSUE', 'STA', 'PRE_NOT_RELIED']

def get_raw_data():
    train_data_url = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/train.json"
    test_data_url = "https://storage.googleapis.com/indianlegalbert/OPEN_SOURCED_FILES/Rhetorical_Role_Benchmark/Data/dev.json"
    train_data_json = requests.get(train_data_url).json()
    test_data_json = requests.get(test_data_url).json()
    return train_data_json, test_data_json

def get_model_input(train_data_json, test_data_json):
    x_train = pd.DataFrame(columns = ["text", "output"])
    x_test = pd.DataFrame(columns = ["text", "output"])

    p = 0
    for i in tqdm(range(len(train_data_json))):
        for j in range(len(train_data_json[i]["annotations"])):
            for k in range(len(train_data_json[i]["annotations"][j]["result"])):
                text = train_data_json[i]["annotations"][j]["result"][k]["value"]["text"]
                output = train_data_json[i]["annotations"][j]["result"][k]["value"]["labels"][0]
                x_train.loc[p] = [text, output]
                p+=1

    p = 0
    for i in tqdm(range(len(test_data_json))):
        for j in range(len(test_data_json[i]["annotations"])):
            for k in range(len(test_data_json[i]["annotations"][j]["result"])):
                text = test_data_json[i]["annotations"][j]["result"][k]["value"]["text"]
                output = test_data_json[i]["annotations"][j]["result"][k]["value"]["labels"][0]
                x_test.loc[p] = [text, output]
                p+=1

    x_train['text'].apply(lambda x: len(x.split(' '))).sum()
    x_test['text'].apply(lambda x: len(x.split(' '))).sum()

    x_data_train = x_train["text"]
    y_data_train = x_train["output"]
    x_data_test = x_test["text"]
    y_data_test = x_test["output"]
    x_data_train = label_sentences(x_data_train, 'Train')
    x_data_test = label_sentences(x_data_test, 'Test')
    all_data = x_data_train + x_data_test
    return x_data_train, y_data_train, x_data_test, y_data_test, all_data

def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors

def get_dbow(all_data, x_data_train, x_data_test):
    
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(60):
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
    
    train_vectors_dbow = get_vectors(model_dbow, len(x_data_train), 300, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(x_data_test), 300, 'Test')
    return train_vectors_dbow, test_vectors_dbow


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled


def get_predictions(train_vectors_dbow, test_vectors_dbow, y_data_train):
    logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter = 70)
    logreg.fit(train_vectors_dbow, y_data_train)
    logreg = logreg.fit(train_vectors_dbow, y_data_train)
    y_pred = logreg.predict(test_vectors_dbow)
    return y_pred


def print_metrics(y_d2v_test, y_pred):
    print('accuracy %s' % metrics.accuracy_score(y_d2v_test, y_pred))
    print('F1 score %s' % metrics.f1_score(y_d2v_test, y_pred, average = "macro"))
    print('precision %s' % metrics.precision_score(y_d2v_test, y_pred, average = "macro"))


def plot_cm(y_data_test, predicted):
    cm = confusion_matrix(y_data_test, predicted)
    cm_df = pd.DataFrame(cm,
                        index = CLASSES,
                        columns = CLASSES)
    plt.figure(figsize=(15,10))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

if __name__ == '__main__':
  train_data_json, test_data_json = get_raw_data()
  x_data_train, y_data_train, x_data_test, y_data_test, all_data = get_model_input(train_data_json, test_data_json)
  train_vectors_dbow, test_vectors_dbow = get_dbow(all_data, x_data_train, x_data_test)
  predicted = get_predictions(train_vectors_dbow, test_vectors_dbow, y_data_train)
  print_metrics(y_data_test, predicted)
  plot_cm(y_data_test, predicted)
