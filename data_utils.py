import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['label'] = df['Category'].map({'ham': 0, 'spam': 1})
    return df

def get_top_features(df, n=20):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['Message'])
    features_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    features_df['label'] = df['label']
    corr_matrix = features_df.corr()
    feature_corr_with_label = corr_matrix['label'].drop('label')
    sorted_features = feature_corr_with_label.abs().sort_values(ascending=False)
    return sorted_features.head(n).index.tolist()
