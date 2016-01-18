def classify(features_train, labels_train, min_sample_split = 2):

    # return a trained decision tree classifer
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=min_sample_split)
    clf = clf.fit(features_train, labels_train)

    return clf

