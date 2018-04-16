import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import classification_report


def load_data(path):
    print('Loading and parsing data...')
    dataset = pd.read_csv(path)
    return dataset.drop(['Unnamed: 0', 'type'], axis=1), dataset['type']


def train_model(X, y):
    print('Training model...')
    adaboost_model = AdaBoostClassifier(n_estimators=100)
    adaboost_model.fit(X, y)
    return adaboost_model


def evaluate_model(model, X, y):
    print('Evaluating model...')
    y_predictions = model.predict(X)

    print(model.score(X, y))
    print(classification_report(y, y_predictions))


def save_model(model, model_path):
    print('Saving model...')
    joblib.dump(model, model_path)


def train_skin_model(dataset_path, model_path):
    dataset_X, dataset_y = load_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)
    save_model(model, model_path)


train_skin_model('./results/features_1.csv', './results/skin_model.pk1')
