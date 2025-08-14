import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path, sample_size=5000, random_state=42):
    columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    data = pd.read_csv(file_path, encoding='ISO-8859-1', names=columns)
    data = data[['sentiment', 'text']].sample(n=sample_size, random_state=random_state)
    data['sentiment'] = data['sentiment'].map({0: 0, 4: 1})
    return train_test_split(data, test_size=0.2, random_state=random_state)
