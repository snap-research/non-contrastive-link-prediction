import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1):
    """Fit a logistic regression model to the data for node classification.
    This is from the official BGRL implementation:
    https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/logistic_regression_eval.py#L9
    """
    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(np.bool)

    # normalize x
    X = normalize(X, norm='l2')

    # set random state to ensure a consistent split
    rng = np.random.RandomState(data_random_seed)

    accuracies = []
    for _ in range(repeat):
        # different random split after each repeat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.8, random_state=rng
        )

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear', max_iter=200)
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=5,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(np.bool)

        test_acc = metrics.accuracy_score(y_test, y_pred)
        accuracies.append(test_acc)
    return accuracies


def do_classification_eval(dataset, embeddings):
    data = dataset[0]
    X = embeddings.weight.cpu().numpy()
    y = data.y.cpu().numpy()
    accs = fit_logistic_regression(X, y)
    return np.mean(accs)
