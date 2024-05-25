import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer


# st.write("""
# # Sole Savant
# Sentiment Analysis and classification for Best Shoe Recommendation
# """)

# data = pd.read_csv('Shoes_Data.csv')
# data.head()

# st.write(data.head())

import streamlit as st
import pandas as pd
import pickle

# Load tokenizer
# with open("model/tokenizer.pkl", "rb") as f:
#     tokenizer = pickle.load(f)

# Load sentiment analysis model
with open("model/sentiment_analysis_model.pkl", "rb") as f:
    sentiment_analysis_model = pickle.load(f)

# Memuat data sepatu
shoe_data = pd.read_csv('data/Shoes_Data.csv')

def main():
    st.title("Sentiment Analysis and Classification for Best Shoe Recommendation")
    st.write("This website provides sentiment analysis for shoe products, allowing users to filter shoes based on brand, price, and reviews.")
    
    # Sidebar untuk filter
    st.sidebar.header("Filter Options")
    brand_filter = st.sidebar.multiselect("Select Brand", options=shoe_data['title'].unique())
    # Menghapus simbol mata uang '₹' dari nilai harga dan mengonversi ke floating-point
    shoe_data['price'] = shoe_data['price'].str.replace('₹', '').astype(float)
    # Menggunakan nilai harga yang sudah dibersihkan dan dikonversi dalam fungsi slider
    price_filter = st.sidebar.slider("Select Price Range", shoe_data['price'].min(), shoe_data['price'].max())
    user_review = st.sidebar.text_area("Enter your review to get sentiment:")

    if st.sidebar.button("Analyze Sentiment"):
        if user_review:
            # Transformasi ulasan pengguna menggunakan TF-IDF vectorizer dan prediksi sentimen
            review_vectorized = tfidf_vectorizer.transform([user_review])
            sentiment_prediction = sentiment_model.predict(review_vectorized)
            sentiment_label = 'Positive' if sentiment_prediction[0] == 1 else 'Negative'
            st.sidebar.write(f"Sentiment: {sentiment_label}")
        else:
            st.sidebar.write("Please enter a review for sentiment analysis.")
    
    # Menyaring data berdasarkan pilihan pengguna
    filtered_data = shoe_data
    if brand_filter:
        filtered_data = filtered_data[filtered_data['title'].isin(brand_filter)]
    filtered_data = filtered_data[filtered_data['price'] <= price_filter]

    st.subheader("Shoe Recommendations")
    st.write(filtered_data)

if __name__ == '__main__':
    main()
