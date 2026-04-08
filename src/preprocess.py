# src/preprocess.py
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

# src/preprocess.py
def load_data(path):
    import pandas as pd

    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ['label', 'id', 'date', 'flag', 'user', 'text']

    df['text'] = df['text'].apply(clean_text)

    # Binary mapping
    df['label'] = df['label'].map({0: 0, 4: 1})

    df = df[df['label'].notnull()]  # IMPORTANT

    return df[['text', 'label']]