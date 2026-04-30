import pandas
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tabulate import tabulate
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

def info_in(y_test, y_pred, name):
    log_regr_accuracy_score2 = accuracy_score(y_test, y_pred)
    log_regr_precision_score2 = precision_score(y_test, y_pred, average='weighted')
    log_regr_f1_score2 = f1_score(y_test, y_pred, average='weighted')
    print('log_regr_accuracy_score ', log_regr_accuracy_score2)
    print('log_regr_precision_score ', log_regr_precision_score2)
    print('log_regr_f1_score ', log_regr_f1_score2)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return [name, log_regr_accuracy_score2, log_regr_precision_score2, log_regr_f1_score2]

def main():
    name = ["Name", "Accuracy", "Precision", "F1-score"]
    result = []
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.4, random_state = 0)

    clf = LogisticRegression(solver='lbfgs', max_iter=3000, random_state=0)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    result.append(info_in(y_test, y_pred, 'Logistic Regression'))

    clf = DecisionTreeClassifier(max_depth=10, random_state=0)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    dec_tree_score2 = clf.score(x_test, y_test)
    print('dec_tree_score ', dec_tree_score2)
    plot_tree(clf)
    result.append(info_in(y_test, y_pred, 'Decision Tree'))

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    rand_frst_score2 = clf.score(x_test, y_test)
    print('dec_tree_score ', rand_frst_score2)
    result.append(info_in(y_test, y_pred, 'Random Forest'))

    clf = KNeighborsClassifier(n_neighbors=3)
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    KNbrs_score2 = clf.score(x_test, y_test)
    print('KNbrs_score ', KNbrs_score2)
    result.append(info_in(y_test, y_pred, 'KNN'))

    clf = SVC(gamma='scale')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    SVC_score2 = clf.score(x_test, y_test)
    print('SVC_score ', SVC_score2)
    result.append(info_in(y_test, y_pred, 'SVC'))

    clf = GaussianNB()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    NB_score2 = clf.score(x_test, y_test)
    print('NB_score ', NB_score2)
    result.append(info_in(y_test, y_pred, 'Naive Bayes'))

    print(tabulate(result, headers=name, tablefmt='grid'))


if __name__ == '__main__':
    main()