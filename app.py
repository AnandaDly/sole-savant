import pandas as pd
import streamlit as st
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import plotly.express as px
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import SimpleRNN

# Memuat data sepatu
shoe_data = pd.read_csv('data/Shoes_Data_Final.csv')

# def app():

def main():
    st.title("Sentiment Analysis and Classification for Best Shoe Recommendation")
    st.write("This website provides sentiment analysis for shoe products, allowing users to filter shoes based on brand, price, and reviews.")
    
    # Sidebar untuk filter
    st.sidebar.header("Filter Options")
    
    brand_options = sorted(shoe_data['merk'].unique())  # Mengurutkan merek secara alfabetis
    brand_filter = st.sidebar.multiselect("Select Brand", options=brand_options)
    
    # Menampilkan opsi untuk pemilihan tipe sepatu
    shoe_type_options = sorted(shoe_data['Shoe Type'].unique())
    shoe_type_filter = st.sidebar.multiselect("Select Shoe Type", options=shoe_type_options)
    
    # Menghapus simbol mata uang '₹' dari nilai harga dan mengonversi ke floating-point
    shoe_data['price_idr'] = shoe_data['price_idr'].astype(float)
    
    # Menggunakan nilai harga yang sudah dibersihkan dan dikonversi dalam fungsi slider
    price_filter = st.sidebar.slider("Select Price Range", float(shoe_data['price_idr'].min()), float(shoe_data['price_idr'].max()))
    
    # Menyaring data berdasarkan pilihan pengguna
    filtered_data = shoe_data
    if brand_filter:
        filtered_data = filtered_data[filtered_data['merk'].isin(brand_filter)]
    if shoe_type_filter:
        filtered_data = filtered_data[filtered_data['Shoe Type'].isin(shoe_type_filter)]
    filtered_data = filtered_data[filtered_data['price_idr'] <= price_filter].reset_index(drop=True)
    
    # Pilih kolom yang ingin ditampilkan
    selected_columns = ['title', 'Shoe Type', 'merk', 'price_idr', 'rating']

    # Filter data berdasarkan kolom yang dipilih
    filtered_data_display = filtered_data[selected_columns]

    # Menambah 1 ke indeks
    filtered_data_display.index += 1
        
    st.subheader("Shoe Recommendations")
    if len(filtered_data) == 0:
        st.write("No data available")
    else:
        st.write(filtered_data_display)
        
        # Membuat dropdown untuk memilih berdasarkan title
        title_options = filtered_data['title'].unique()
        selected_title = st.selectbox("Select a title to view details", options=title_options)
        
        # Tombol untuk menampilkan detail
        if st.button("View Details"):
            st.session_state['viewing_details'] = False
        if 'show_reviews' not in st.session_state:
            st.session_state['show_reviews'] = False
        
        # Tombol untuk kembali ke visualisasi
        if 'viewing_details' in st.session_state and st.session_state['viewing_details']:
            selected_data = filtered_data[filtered_data['title'] == selected_title].iloc[0]
            
            st.subheader(f"Details for {selected_title}")
            st.write(f"**Brand:** {selected_data['merk']}")
            st.write(f"**Deskripsi Produk:** {selected_data['product_description']}")
            st.write(f"**Rating:** {selected_data['rating']}")
    
            # Button to toggle showing reviews
            if st.button('Show/Hide Reviews'):
                st.session_state['show_reviews'] = not st.session_state['show_reviews']
    
            # Menampilkan review dan review rating jika tombol ditekan
            if st.session_state['show_reviews']:
                for i in range(1, 11):
                    review_col = f'review_{i}'
                    rating_col = f'review_rating_{i}'
                    if review_col in selected_data and rating_col in selected_data:
                        rating_value = selected_data[rating_col]
                        review_value = selected_data[review_col]
                        if pd.notna(rating_value) and pd.notna(review_value):
                            st.write(f"{rating_value}")
                            st.write(f"{review_value}")
                            st.markdown('---')

            
            if st.button("Back to Visualizations"):
                st.session_state['viewing_details'] = False
        else:
            # Menghitung persentase rating baik dan buruk
            good_ratings = filtered_data[filtered_data['rating'].apply(lambda x: float(re.search(r'(\d+\.\d+)', x).group()) > 3.0)]
            bad_ratings = filtered_data[filtered_data['rating'].apply(lambda x: float(re.search(r'(\d+\.\d+)', x).group()) <= 3.0)]
            total_good_percent = (len(good_ratings) / len(filtered_data)) * 100
            total_bad_percent = (len(bad_ratings) / len(filtered_data)) * 100
            
            # Menghitung rata-rata harga untuk sepatu dengan rating baik dan buruk
            avg_price_good_ratings = good_ratings['price_idr'].mean()
            avg_price_bad_ratings = bad_ratings['price_idr'].mean()

            # Visualisasi menggunakan plotly
            fig = px.pie(names=['Good Ratings', 'Bad Ratings'], values=[total_good_percent, total_bad_percent], title='Rating Distribution')
            st.plotly_chart(fig)
            
            # Membuat bar chart untuk perbandingan harga
            fig2 = px.bar(x=['Good Ratings', 'Bad Ratings'], y=[avg_price_good_ratings, avg_price_bad_ratings],
                          labels={'x': 'Rating Category', 'y': 'Average Price'},
                          title='Average Price Comparison Between Good and Bad Ratings')
            st.plotly_chart(fig2)

if __name__ == '__main__':
    main()
