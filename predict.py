def predict_messages(clf, vectorizer, messages):
    X_new = vectorizer.transform(messages)
    return clf.predict(X_new)
