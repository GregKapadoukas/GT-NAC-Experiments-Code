from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsClassifier


def knn_classifier(data_train, categories_train, data_test, categories_test, k):
    if data_train.shape[0] < k:
        print(
            f"Overriding k to {data_train.shape[0]} because there are not enough samples"
        )
        k = data_train.shape[0]
    knnc = KNeighborsClassifier(n_neighbors=k)
    knnc.fit(data_train, categories_train)
    categories_pred = knnc.predict(data_test)
    accuracy = accuracy_score(categories_test, categories_pred)
    macro_precision, macro_recall, macro_fscore, macro_support = score(
        categories_test, categories_pred, average="macro"
    )
    print(f"Accuracy: {accuracy}")
    print(f"Macro-Precision: {macro_precision}")
    print(f"Macro-Recall: {macro_recall}")
    print(f"Macro-F-Score: {macro_fscore}")
    print(classification_report(categories_test, categories_pred, zero_division=1))
    return {
        "Accuracy": accuracy,
        "Macro-Precision": macro_precision,
        "Macro-Recall": macro_recall,
        "Macro-F-Score": macro_fscore,
    }
