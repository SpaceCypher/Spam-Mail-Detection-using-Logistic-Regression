from data_utils import load_data, get_top_features
from model import train_and_evaluate
from predict import predict_messages

if __name__ == "__main__":
    df = load_data('spam.csv')
    top_features = get_top_features(df, n=20)
    clf, vectorizer = train_and_evaluate(df, top_features)
    # Example 
    messages = ["Free entry in a competition to win a $1000 gift card!","Congratulations! You've won a free ticket to Bahamas!", "Hey, how are you doing today?"]
    predictions = predict_messages(clf, vectorizer, messages)
    print(predictions) 
        # 1 = Spam ; 0 = Ham/Legitmate Message