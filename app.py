import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import SimpleRNN

import matplotlib.pyplot as plt
import re
import plotly.express as px

# Memuat data sepatu
shoe_data = pd.read_csv('data/Shoes_data_edit1.csv')

def main():
    st.title("Sentiment Analysis and Classification for Best Shoe Recommendation")
    st.write("This website provides sentiment analysis for shoe products, allowing users to filter shoes based on brand, price, and reviews.")
    
    # Sidebar untuk filter
    st.sidebar.header("Filter Options")
    
    brand_options = sorted(shoe_data['merk'].unique())  # Mengurutkan merek secara alfabetis
    brand_filter = st.sidebar.multiselect("Select Brand", options=brand_options)
    
    # Menampilkan opsi untuk pemilihan tipe sepatu
    shoe_type_options = sorted(shoe_data['shoe type'].unique())
    shoe_type_filter = st.sidebar.multiselect("Select Shoe Type", options=shoe_type_options)
    
    # Menghapus simbol mata uang '₹' dari nilai harga dan mengonversi ke floating-point
    shoe_data['price'] = shoe_data['price'].str.replace('₹', '').astype(float)
    
    # Menggunakan nilai harga yang sudah dibersihkan dan dikonversi dalam fungsi slider
    price_filter = st.sidebar.slider("Select Price Range", float(shoe_data['price'].min()), float(shoe_data['price'].max()))
    
    user_review = st.sidebar.text_area("Enter your review to get sentiment:")

    if st.sidebar.button("Analyze Sentiment"):
        if user_review:
            # Transformasi ulasan pengguna menggunakan tokenizer dan prediksi sentimen
            review_sequence = tokenizer.texts_to_sequences([user_review])
            review_padded = pad_sequences(review_sequence, maxlen=100)
            sentiment_prediction = sentiment_analysis_model.predict(review_padded)
            sentiment_label = 'Positive' if sentiment_prediction[0] >= 0.5 else 'Negative'
            st.sidebar.write(f"Sentiment: {sentiment_label}")
        else:
            st.sidebar.write("Please enter a review for sentiment analysis.")
    
    # Menyaring data berdasarkan pilihan pengguna
    filtered_data = shoe_data
    if brand_filter:
        filtered_data = filtered_data[filtered_data['merk'].isin(brand_filter)]
    if shoe_type_filter:
        filtered_data = filtered_data[filtered_data['shoe type'].isin(shoe_type_filter)]
    filtered_data = filtered_data[filtered_data['price'] <= price_filter].reset_index(drop=True)
    
    # Menambah 1 ke indeks
    filtered_data.index += 1
    
    st.subheader("Shoe Recommendations")
    if len(filtered_data) == 0:
        st.write("No data available")
    else:
        st.write(filtered_data)
        
        # Menghitung persentase rating baik dan buruk
        good_ratings = filtered_data[filtered_data['rating'].apply(lambda x: float(re.search(r'(\d+\.\d+)', x).group()) > 3.0)]
        bad_ratings = filtered_data[filtered_data['rating'].apply(lambda x: float(re.search(r'(\d+\.\d+)', x).group()) <= 3.0)]
        total_good_percent = (len(good_ratings) / len(filtered_data)) * 100
        total_bad_percent = (len(bad_ratings) / len(filtered_data)) * 100

        st.write(f"Percentage of good ratings: {total_good_percent:.2f}%")
        st.write(f"Percentage of bad ratings: {total_bad_percent:.2f}%")
        
        # Visualisasi menggunakan plotly
        fig = px.pie(names=['Good Ratings', 'Bad Ratings'], values=[total_good_percent, total_bad_percent], title='Rating Distribution')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
