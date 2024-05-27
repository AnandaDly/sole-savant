import streamlit as st
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load model
with open('model/sentiment_analysis_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to clean text data
def clean_text(text):
    # Remove emojis
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load data
df = pd.read_csv('data/Shoes_Data_Final.csv')

# Clean reviews
df['cleaned_reviews'] = df['reviews'].apply(clean_text)

# Sidebar for user input
st.sidebar.header('Filter')
selected_brand = st.sidebar.selectbox('Select Brand', df['merk'].unique())
price_range = st.sidebar.slider('Select Price Range', int(df['price_idr'].min()), int(df['price_idr'].max()), (int(df['price_idr'].min()), int(df['price_idr'].max())))
shoe_type = st.sidebar.selectbox('Select Shoe Type', df['Shoe Type'].unique())

# Filter data based on user input
filtered_data = df[(df['merk'] == selected_brand) & (df['price_idr'] >= price_range[0]) & (df['price_idr'] <= price_range[1]) & (df['Shoe Type'] == shoe_type)]

st.title('Shoe Sentiment Analysis')

if not filtered_data.empty:
    st.write(f'Found {len(filtered_data)} shoes matching your criteria.')
    for index, row in filtered_data.iterrows():
        st.subheader(row['title'])
        st.write(f"**Brand:** {row['merk']}")
        st.write(f"**Type:** {row['Shoe Type']}")
        st.write(f"**Price:** {row['price_idr']}")
        st.write(f"**Total Reviews:** {row['total_reviews']}")
        st.write(f"**Rating:** {row['rating']}")
        st.write(f"**Description:** {row['product_description']}")
        st.write("**Reviews:**")
        st.write(row['reviews'])

        # Predict sentiment
        sentiment = model.predict([['cleaned_reviews']])
        sentiment_percentage = sentiment.mean() * 100
        st.write(f"**Sentiment Analysis:** {sentiment_percentage:.2f}% positive")
else:
    st.write('No shoes found matching your criteria.')

if __name__ == '__main__':
    st.run()
