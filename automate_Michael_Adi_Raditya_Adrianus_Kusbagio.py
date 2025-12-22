import pandas as pd
import numpy as np
import re
import string
import requests
import csv
from io import StringIO, BytesIO
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class TextPreprocessor:
    def __init__(self):
        self._setup_nltk()
        self.stop_words = self._get_stopwords()
        self.lexicon_positive, self.lexicon_negative = self._get_lexicons()
        self.kamus_slang = self._load_remote_slang()

    def _setup_nltk(self):
        # Memastikan resource terunduh hanya sekali
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)

    def _get_stopwords(self):
        stop_words = set(stopwords.words('indonesian'))
        stop_words.update(set(stopwords.words('english')))
        return stop_words

    def _load_remote_slang(self):
        # URL kamus slang yang Anda berikan
        url = "https://github.com/MichaelAdi434/Project-Analisis-Sentimen/raw/d21c7566deca33e2871f160f19728f39d5fd273d/kamuskatabaku.xlsx"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Menggunakan BytesIO karena file adalah binary (.xlsx)
                df_slang = pd.read_excel(BytesIO(response.content))
                # Sesuai notebook: kolom kata1 sebagai key dan kata2 sebagai value
                return dict(zip(df_slang['kata1'], df_slang['kata2']))
            else:
                return {}
        except Exception as e:
            print(f"Error saat memuat kamus: {e}")
            return {}

    def _get_lexicons(self):
        pos = {}
        neg = {}
        # Sesuai notebook: menggunakan requests untuk fetch data eksternal
        try:
            res_pos = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_positive.csv')
            if res_pos.status_code == 200:
                reader = csv.reader(StringIO(res_pos.text))
                pos = {row[0]: int(row[1]) for row in reader}
            
            res_neg = requests.get('https://raw.githubusercontent.com/angelmetanosaa/dataset/main/lexicon_negative.csv')
            if res_neg.status_code == 200:
                reader = csv.reader(StringIO(res_neg.text))
                neg = {row[0]: int(row[1]) for row in reader}
        except Exception as e:
            print(f"Error lexicon: {e}")
        
        return pos, neg

    def clean_text(self, text):
        if not isinstance(text, str): return ""
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#[A-Za-z0-9]+', '', text)
        text = re.sub(r'RT[\s]', '', text)
        text = re.sub(r"http\S+", '', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = text.replace('\n', ' ')
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.strip(' ')
        return text.lower() # Sekaligus case folding sesuai notebook

    def handle_slang(self, text):
        return ' '.join([self.kamus_slang.get(kata, kata) for kata in text.split()])

    def process_tokens(self, text):
        tokens = word_tokenize(text)
        # Filtering teks sesuai urutan notebook
        return [t for t in tokens if t not in self.stop_words]

    def get_sentiment(self, tokens):
        # Logika scoring lexicon sesuai notebook
        score = 0
        for t in tokens:
            if t in self.lexicon_positive:
                score += self.lexicon_positive[t]
            if t in self.lexicon_negative:
                score += self.lexicon_negative[t]
        return 'positive' if score >= 0 else 'negative'

    def run_pipeline(self, input_path, output_path):
        # 1. Load Data
        df = pd.read_csv(input_path)
        
        # 2. Preprocessing (Drop kolom & Duplikat)
        # Perbaikan: dropna digunakan untuk baris, bukan kolom
        df = df.drop(columns=['reviewCreatedVersion', 'replyContent', 'repliedAt', 'appVersion'], errors='ignore')
        df = df.dropna(subset=['content']).drop_duplicates()
        
        # 3. Pipeline Transformation
        df['clean_text'] = df['content'].apply(self.clean_text)
        df['normalized'] = df['clean_text'].apply(self.handle_slang)
        df['tokens'] = df['normalized'].apply(self.process_tokens)
        df['text_akhir'] = df['tokens'].apply(lambda x: ' '.join(x))
        df['label'] = df['tokens'].apply(self.get_sentiment)
        
        # 4. Save Ready-to-Train Dataset
        # Memastikan folder output tersedia
        df[['text_akhir', 'label']].to_csv(output_path, index=False)
        print(f"Preprocessing selesai! Data disimpan ke {output_path}")

if __name__ == "__main__":
    prep = TextPreprocessor()
    # Path disesuaikan untuk workflow GitHub Actions
    prep.run_pipeline('data/ulasan_apk.csv', 'preprocessing/processed_data.csv')
