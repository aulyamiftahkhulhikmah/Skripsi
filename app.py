import numpy as np
import pandas as pd
import streamlit as st
import pickle
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')


st.header('Analisis Sentimen Ulasan Pengguna Aplikasi Layanan Kesehatan menggunakan SVM dan SMOTE')

text_input = st.text_area("Masukkan Teks")
submit = st.button("Submit", type="primary")

if submit:
    if text_input:
        df_mentah = pd.DataFrame({'Ulasan':[text_input]})
        def remove_punctuation(text):
            # Memeriksa apakah input adalah string, jika tidak, ubah menjadi string
            if not isinstance(text, str):
                text = str(text)
            #menghilangkan url
            text = re.sub(r'https?:\/\/\S+','',text)
            #menghilangkan mention, link, hastag
            text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
            #menghilangkan karakter byte (b')
            text = re.sub(r'(b\'{1,2})',"", text)
            #menghilangkan yang bukan huruf
            text = re.sub('[^a-zA-Z]', ' ', text)
            #menghilangkan digit angka
            text = re.sub(r'\d+', '', text)
            #menghilangkan tanda baca
            text = text.translate(str.maketrans("","",string.punctuation))
            #menghilangkan whitespace berlebih
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def lower_case(text):
            lower = text.lower()
            return lower

        #Tokenization
        def tokenization(text):
            text = re.split('\W+', text)
            return text

        # Mendefinisikan daftar stopwords untuk bahasa Indonesia menggunakan nltk
        stopword = nltk.corpus.stopwords.words('indonesian')

        # Fungsi untuk menghapus stopwords dari teks
        def remove_stopwords(text):
            # Menggunakan list comprehension untuk hanya menyimpan kata-kata yang bukan stopwords
            text = [word for word in text if word not in stopword]
            return text

        # Membuat instance dari StemmerFactory
        factory = StemmerFactory()
        # Menggunakan factory untuk membuat stemmer
        stemmer = factory.create_stemmer()

        # Fungsi untuk melakukan stemming pada teks
        def stemming(text):
            return stemmer.stem(text) 

        # st.write("Data Mentah")
        # st.write(df_mentah['Ulasan'])

        df_mentah['cleaning'] = df_mentah['Ulasan'].apply(remove_punctuation)
        # st.write("cleaning")
        # st.write(df_mentah['cleaning'])

        # st.write("lower case")
        df_mentah['lower'] = df_mentah['cleaning'].apply(lower_case)
        # st.write(df_mentah['lower'])

        # st.write("Tokenisasi")
        df_mentah['tokenisasi'] = df_mentah['lower'].apply(tokenization)
        # st.write(df_mentah['tokenisasi'])
        
        # st.write("Stopword Removal")
        df_mentah['stopword'] = df_mentah['tokenisasi'].apply(remove_stopwords)
        # st.write(df_mentah['stopword'])

        # st.write("Stemming")
        df_mentah['stem'] = (df_mentah['stopword'].astype(str)).apply(stemming)
        # st.write(df_mentah['stem'])

        path_tfidf = 'Model/tfidf.pkl'
        with open(path_tfidf, 'rb') as f:
            tfidf = pickle.load(f)

        # Transform the 'stem' column using the loaded TfidfVectorizer
        doc_vectors = tfidf.transform(df_mentah['stem']).toarray()
        df_result = pd.DataFrame(doc_vectors, columns=tfidf.get_feature_names_out())
        # st.write(df_result)

        #load model
        path_svm = 'Model/UjiCoba2_SVMSMOTE k3 C10.pkl'
        with open (path_svm, 'rb') as f:
            svm_model = pickle.load(f)

        y_pred = svm_model.predict(df_result)
        if y_pred == 0:
            st.write(f'Ulasan **"{text_input}"** Memiliki Sentimen : **Negatif**')
        if y_pred == 1:
            st.write(f'Ulasan **"{text_input}"** Memiliki Sentimen : **Positif**')
    else:
        st.write('Kamu Belum Memasukkan Teks😊')

