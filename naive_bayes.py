#import all the required packages/libraries
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import requests

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


    x_data_train = x_train["text"]
    y_data_train = x_train["output"]
    x_data_test = x_test["text"]
    y_data_test = x_test["output"]
    return x_data_train, y_data_train, x_data_test, y_data_test

def get_predictions(x_data_train, x_data_test):
    cv = CountVectorizer()
    xtrain_dtm = cv.fit_transform(x_data_train)
    xtest_dtm=cv.transform(x_data_test)
    print('\n The words or Tokens in the text documents \n')
    print(cv.get_feature_names_out())
    df=pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names_out())


    clf = MultinomialNB().fit(xtrain_dtm,y_data_train)
    predicted = clf.predict(xtest_dtm)
    train_predicted = clf.predict(xtrain_dtm)
    return train_predicted, predicted

def print_metrics(y_data_train, train_predicted, y_data_test, predicted):
    print("Train Data")
    print('\n Accuracy of the classifier is',metrics.accuracy_score(y_data_train,train_predicted))
    print('\n The value of Precision', metrics.precision_score(y_data_train,train_predicted, average= "macro"))
    print('\n The value of macro F1 score', metrics.f1_score(y_data_train,train_predicted, average= "macro"))

    print("Test Data")
    print('\n Accuracy of the classifier is',metrics.accuracy_score(y_data_test,predicted))
    print('\n The value of Precision', metrics.precision_score(y_data_test,predicted, average= "macro"))
    print('\n The value of Recall', metrics.recall_score(y_data_test,predicted, average= "macro"))
    print('\n The value of macro F1 score', metrics.f1_score(y_data_test,predicted, average= "macro"))

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
  x_data_train, y_data_train, x_data_test, y_data_test = get_model_input(train_data_json, test_data_json)
  train_predicted, predicted = get_predictions(x_data_train, x_data_test)
  print_metrics(y_data_train, train_predicted, y_data_test, predicted)
  plot_cm(y_data_test, predicted)
