import pandas as pd
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import SimpleRNN

# # Load tokenizer
# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'keras.src.preprocessing.text':
#             module = 'tensorflow.keras.preprocessing.text'
#         return super().find_class(module, name)

# with open("model/tokenizer.pkl", "rb") as f:
#     tokenizer = CustomUnpickler(f).load()

# # Load the model with custom objects
# sentiment_analysis_model = tf.keras.models.load_model("model/sentiment_analysis_model.h5", custom_objects={'Orthogonal': tf.keras.initializers.Orthogonal}, compile=False)
# # sentiment_analysis_model = tf.keras.models.load_model("model/sentiment_analysis_model.h5", custom_objects={'SimpleRNN': SimpleRNN}, compile=False)
# # sentiment_analysis_model = tf.keras.models.load_model("model/sentiment_analysis_model.h5", compile=False)

# # Compile the model
# sentiment_analysis_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Memuat data sepatu
shoe_data = pd.read_csv('data/Shoes_data_edit1.csv')

def main():
    st.title("Sentiment Analysis and Classification for Best Shoe Recommendation")
    st.write("This website provides sentiment analysis for shoe products, allowing users to filter shoes based on brand, price, and reviews.")
    
    # Sidebar untuk filter
    st.sidebar.header("Filter Options")
    brand_filter = st.sidebar.multiselect("Select Brand", options=shoe_data['title'].unique())
    
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
        filtered_data = filtered_data[filtered_data['title'].isin(brand_filter)]
    filtered_data = filtered_data[filtered_data['price'] <= price_filter]

    st.subheader("Shoe Recommendations")
    st.write(filtered_data)

if __name__ == '__main__':
    main()
