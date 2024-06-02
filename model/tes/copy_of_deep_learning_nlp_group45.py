# -*- coding: utf-8 -*-
"""Copy of Deep_Learning_NLP_Group45.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_pConTyYxTVKN8-2CiFrWa-q7-PAkb-g

# Kampus Merdeka 6: IBM & Skilvul
# Artificial Intelligence Phase Challenge

## Kelompok: 45
## Anggota

*   Alif M. Anwar Tambunan
*   Asep Nugraha
*   Egi Saputra
*   RIZKY ANANDA ALAM SYAH DAULAY
*   Azhar Syahid

# Problem Definition
Dalam proyek ini, kita bertujuan untuk menganalisis sentimen ulasan sepatu menggunakan teknik pemrosesan bahasa alami (NLP). Secara spesifik, kita ingin mengklasifikasikan ulasan sebagai positif atau negatif berdasarkan isi teksnya. Tugas ini sangat penting bagi bisnis untuk memahami opini pelanggan dan meningkatkan produk dan layanan mereka.

## Latar Belakang
Dengan meningkatnya e-commerce, ulasan online telah menjadi sumber informasi yang sangat penting bagi pelanggan untuk membuat keputusan pembelian yang informasi. Oleh karena itu, menganalisis sentimen dalam ulasan telah menjadi area penelitian yang signifikan dalam NLP. Dalam proyek ini, kita fokus pada menganalisis sentimen dalam ulasan sepatu, yang merupakan pasar yang populer dan kompetitif.

## Tujuan Penelitian
Tujuan dari penelitian ini adalah untuk mengembangkan dan mengevaluasi kinerja dari berbagai model deep learning untuk analisis sentimen dalam ulasan sepatu. Secara spesifik, kita bertujuan untuk:

- Membandingkan kinerja dari empat model deep learning yang populer: LSTM, RNN, GRU, dan CNN
- Mengidentifikasi model yang paling baik untuk analisis sentimen dalam ulasan sepatu
- Mengevaluasi efektivitas dari setiap model dalam menangkap pola sentimen dalam data teks

## Pertanyaan kunci
- Model deep learning mana yang paling baik untuk analisis sentimen dalam ulasan sepatu?
- Bagaimana kinerja dari setiap model dibandingkan dengan yang lain?
- Dapatkah kita mencapai akurasi yang tinggi dalam analisis sentimen menggunakan model deep learning?

## Data yang akan dipakai
* Nama Dataset : Men_Women_Shoes_Reviews,
* Sumber Dataset : Kaagle,
* Deskripsi Data : Dataset ini berisi informasi tentang sepatu yang meliputi berbagai atribut seperti judul (Title), harga (Price), rating (Rating), deskripsi produk (Product Description), dan tipe sepatu (Shoe Type). Rating merepresentasikan total penilaian yang diberikan oleh pengguna, dengan rentang nilai dari 1 hingga 5. Total Reviews menunjukkan jumlah ulasan yang diterima oleh sepatu tersebut dari pengguna. Kolom Reviews berisi hingga 10 ulasan tentang sepatu, yang dipisahkan oleh tanda '||'. Setiap ulasan memiliki kolom Reviews Rating yang mencantumkan penilaian yang sesuai untuk ulasan tersebut, juga dipisahkan oleh tanda '||'. Dataset ini merupakan hasil dari pengumpulan ulasan produk dari sumber yang tidak disebutkan.

## Jenis Masalah Neural Network
Natural Language Processing (NLP).

## Model
model yang di uji :
1. LSTM (Long Short-Term Memory).
2. RNN (Recurrent Neural Network)
3. GRU (Gated Recurrent Unit)
4. CNN (Convolutional Neural Network)

# Preparation | Persiapan
## Import Libraries
"""

# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

"""## Get Data | Mendapatkan Data"""

# get data
df = pd.read_csv('data/Shoes_Data.csv')

"""## Explore Data (EDA) | Eksplorasi Data"""

#from google.colab import drive
#drive.mount('/content/drive')|

# check data
df.head()

# describe data
df.describe()

"""histogram untuk melihat distribusi rating produk pada dataset."""

# perform data visualization
plt.hist(df['reviews_rating'], bins=5, edgecolor='black')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

"""Visualisasikan panjang ulasan untuk mendapatkan pemahaman tentang panjang ulasan yang ada dalam dataset."""

df['review_length'] = df['product_description'].apply(len)
plt.hist(df['review_length'], bins=30, edgecolor='black')
plt.title('Distribution of Review Length')
plt.xlabel('Review Length')
plt.ylabel('Count')
plt.show()

# Menggunakan word cloud untuk melihat kata-kata yang paling sering muncul dalam deskripsi produk.

text = ' '.join(df['product_description'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Word Cloud of Product Descriptions')
plt.axis('off')
plt.show()

"""## Preprocess Data | Proses Awal Data"""

# melakukan ekstraksi string ke numerik pada kolom rating agar dapat menjadi acuan kolom "sentiment" yang baru.

# Ekstraksi nilai rating dari kolom 'rating'
def extract_rating(text):
    match = re.search(r'\d+\.\d+', text)  # Gunakan regex untuk menemukan nilai desimal dalam teks
    if match:
        return float(match.group())
    else:
        return None  # Jika tidak ada nilai yang ditemukan, kembalikan None

# Terapkan fungsi extract_rating ke setiap entri dalam kolom 'rating'
df['numeric_rating'] = df['rating'].apply(extract_rating)

# Konversi nilai rating ke dalam format numerik
df['numeric_rating'] = pd.to_numeric(df['numeric_rating'])

# Tentukan batas sentimen
threshold = 3

print(df[['rating', 'numeric_rating']].head(10))

# Buat kolom sentimen berdasarkan nilai rating
df['sentiment'] = df['numeric_rating'].apply(lambda x: 1 if x >= threshold else 0)

df.head()

# Drop kolom yang tidak digunakan
df.drop(['title', 'price', 'total_reviews', 'reviews', 'reviews_rating', 'review_length'], axis=1, inplace=True)

df.head()

# Mengecek baris yang mengandung nilai null
null_rows = df[df.isnull().any(axis=1)]

# Menampilkan baris yang mengandung nilai null
print(null_rows)

# split data
X = df['product_description']
y = df['sentiment']

# Membagi data menjadi data pelatihan dan data uji dengan perbandingan 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Tokenisasi

# Tentukan parameter tokenizer
vocab_size = 10000  # Jumlah kata yang akan diambil
embedding_dim = 100  # Dimensi embedding
max_length = 100  # Panjang maksimum dari setiap sequence
trunc_type = 'post'  # Jika teks lebih panjang dari max_length, potong dari akhir
padding_type = 'post'  # Jika teks lebih pendek dari max_length, tambahkan padding di akhir
oov_token = '<OOV>'  # Tanda untuk kata-kata yang tidak ada dalam vocab

# Tokenisasi teks pelatihan
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

# Ubah teks menjadi sequence
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Tokenisasi teks uji
X_test_sequences = tokenizer.texts_to_sequences(X_test)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

"""# Model Training | Pelatihan Model

Sesuai tujuan penelitian dan karakteristik data, masalah neural network pada project kami ini adalah Natural Languange Processing sehingga kami mencoba untuk melakukan pelatihan dengan model-model berikut :
1. LSTM,
LSTM adalah jenis khusus dari jaringan rekurent (RNN) yang dirancang untuk mengatasi masalah vanishing gradient, yang sering terjadi saat melatih jaringan rekurent yang lebih dalam.
LSTM memiliki struktur internal yang kompleks dengan gerbang masuk (input gate), gerbang keluar (output gate), dan gerbang lupa (forget gate), yang memungkinkan untuk menjaga dan mengatur aliran informasi dalam jangka waktu yang lebih lama.

2. RNN,
RNN adalah jenis jaringan saraf tiruan yang memiliki hubungan antara neuron dalam satu lapisan dan juga memiliki hubungan siklik atau rekuren. Artinya, output dari satu langkah waktu menjadi bagian dari input untuk langkah waktu berikutnya.

3. GRU,
GRU adalah alternatif yang lebih sederhana dari LSTM yang juga dirancang untuk mengatasi masalah vanishing gradient dalam jaringan rekurent.
Meskipun strukturnya lebih sederhana daripada LSTM, GRU tetap efektif dalam mengingat informasi jangka panjang dalam data berurutan.
GRU memiliki dua gerbang: gerbang reset (reset gate) yang mengontrol berapa banyak informasi lama yang harus dilupakan, dan gerbang update (update gate) yang mengontrol seberapa banyak informasi baru yang harus diperbarui.

4. CNN,
CNN adalah jenis jaringan saraf tiruan yang dirancang untuk memproses data spasial, seperti gambar. CNN menggunakan filter yang disebut "kernel" untuk mendeteksi pola dalam data, yang memungkinkan model untuk menangkap fitur-fitur yang relevan dalam data. Dalam proyek ini, kita menggunakan CNN untuk menganalisis sentimen dalam ulasan sepatu.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# build model
model_lstm = Sequential()

# add layers
model_lstm.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model_lstm.add(LSTM(units=64))
model_lstm.add(Dropout(0.5))  # Dropout dengan tingkat dropout 0.5
model_lstm.add(Dense(1, activation='sigmoid'))

# Compile model
optimizer = Adam(learning_rate=0.0001)  # Atur learning rate sesuai kebutuhan Anda
model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Stop training if val_loss does not improve for 10 consecutive epochs

# Fit / Run model with Early Stopping callback
history = model_lstm.fit(X_train_padded, y_train, epochs=100, validation_data=(X_test_padded, y_test), verbose=2, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model_lstm.evaluate(X_test_padded, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# build model
model_rnn = Sequential()

# add layers
model_rnn.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model_rnn.add(SimpleRNN(units=64))
model_rnn.add(Dropout(0.5))  # Dropout dengan tingkat dropout 0.5
model_rnn.add(Dense(1, activation='sigmoid'))

# Compile model
optimizer = Adam(learning_rate=0.001)  # Atur learning rate sesuai kebutuhan Anda
model_rnn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Stop training if val_loss does not improve for 10 consecutive epochs

# Fit / Run model with Early Stopping callback
history = model_rnn.fit(X_train_padded, y_train, epochs=100, validation_data=(X_test_padded, y_test), verbose=2, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model_rnn.evaluate(X_test_padded, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# build model
model_gru = Sequential()

# add layers
model_gru.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model_gru.add(GRU(units=64))
model_gru.add(Dropout(0.5))  # Dropout dengan tingkat dropout 0.5
model_gru.add(Dense(1, activation='sigmoid'))

# Compile model
optimizer = Adam(learning_rate=0.001)
model_gru.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Fit / Run model with Early Stopping callback
history = model_gru.fit(X_train_padded, y_train, epochs=100, validation_data=(X_test_padded, y_test), verbose=2, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model_gru.evaluate(X_test_padded, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# build model
model_cnn = Sequential()

# add layers
embedding_dim = 100
filter_sizes = [3, 4, 5]
num_filters = 128
dropout_rate = 0.5

model_cnn.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
for filter_size in filter_sizes:
    model_cnn.add(Conv1D(num_filters, filter_size, activation='relu'))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dropout(dropout_rate))
model_cnn.add(Dense(1, activation='sigmoid'))

# Compile model
optimizer = Adam(learning_rate=0.001)
model_cnn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Fit / Run model with Early Stopping callback
history = model_cnn.fit(X_train_padded, y_train, epochs=100, validation_data=(X_test_padded, y_test), verbose=2, callbacks=[early_stopping])

# Evaluate model
loss, accuracy = model_cnn.evaluate(X_test_padded, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# print model summary
print(model_lstm.summary())

# print model summary
print(model_rnn.summary())

# print model summary
print(model_gru.summary())

# print model summary
print(model_cnn.summary())

"""### Penjelasan tentang Hyperparameter yang dipilih

Untuk setiap model, kami menyetel hyperparameter untuk menemukan kombinasi parameter terbaik yang akan menghasilkan akurasi tertinggi. Berikut ini adalah hyperparameter yang kami setel untuk setiap model:

- LSTM: Kami menggunakan embedding layer dengan dimensi 100, diikuti oleh LSTM layer sebesar 64 unit. Kami juga menambahkan dropout layer dengan rate 0,5 untuk mencegah overfitting.

- RNN: Kami menggunakan embedding layer dengan dimensi 100, diikuti oleh layer RNN dengan 64 unit. Kami juga menambahkan layer dropout dengan rate 0,5 untuk mencegah overfitting.

- GRU: Kami menggunakan layer penyisipan dengan dimensi 100, diikuti oleh layer GRU dengan 64 unit. Kami juga menambahkan layer dropout dengan rate 0,5 untuk mencegah overfitting.

- CNN: Kami menggunakan layer embedding dengan dimensi 100, diikuti oleh serangkaian layer konvolusi dengan ukuran filter 3, 4, dan 5. Kami juga menambahkan layer global max pooling, layer dropout dengan rate 0.5, dan layer dense dengan fungsi aktivasi sigmoid.

Untuk setiap model, kami menggunakan optimizer Adam dengan learning rate 0.001 dan binary cross-entropy sebagai fungsi kerugian. Kami juga menggunakan penghentian awal dengan nilai patience 10 untuk mencegah overfitting.

Alasan kami memilih hyperparameter ini adalah karena hyperparameter ini telah terbukti bekerja dengan baik pada tugas-tugas NLP sebelumnya. Dengan menyamakan nilai untuk setiap hyperparameter membuat kami dapat mengetahui kinerja model mana yang terbaik secara objektif untuk analisis sentiment yang kami lakukan.

# Model Evaluation | Evaluasi Model
"""

# Evaluasi model LSTM
loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test_padded, y_test)

# Evaluasi model RNN
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test_padded, y_test)

# Evaluasi model CNN
loss_gru, accuracy_gru = model_gru.evaluate(X_test_padded, y_test)

# Evaluasi model CNN
loss_cnn, accuracy_cnn = model_cnn.evaluate(X_test_padded, y_test)

# Membandingkan hasil evaluasi dari kedua model
print("Model LSTM:")
print("Test Loss:", loss_lstm)
print("Test Accuracy:", accuracy_lstm)

print("\nModel RNN:")
print("Test Loss:", loss_rnn)
print("Test Accuracy:", accuracy_rnn)

print("\nModel GRU:")
print("Test Loss:", loss_gru)
print("Test Accuracy:", accuracy_gru)

print("\nModel CNN:")
print("Test Loss:", loss_cnn)
print("Test Accuracy:", accuracy_cnn)

# Memilih model terbaik berdasarkan akurasi dan loss
best_accuracy = max(accuracy_lstm, accuracy_rnn,accuracy_gru, accuracy_cnn)
best_loss = min(loss_lstm, loss_rnn, loss_gru, loss_cnn)

if  best_accuracy == accuracy_lstm:
    best_model = "LSTM"
    best_accuracy = accuracy_lstm
    best_loss = loss_lstm
elif best_accuracy == accuracy_rnn:
    best_model = "RNN"
    best_accuracy = accuracy_rnn
    best_loss = loss_rnn
elif best_accuracy == accuracy_gru:
    best_model = "GRU"
    best_accuracy = accuracy_gru
    best_loss = loss_gru
else:
    best_model = "CNN"
    best_accuracy = accuracy_cnn
    best_loss = loss_cnn

print("\nModel terbaik adalah:", best_model)
print("Dengan akurasi:", best_accuracy)
print("Dengan loss:", best_loss)

"""### Metriks yang dipakai

Pertama, loss function. Loss function adalah ukuran yang menunjukkan seberapa baik model memperkirakan label target. Dalam konteks analisis sentimen terhadap produk sepatu, loss function membantu kita memahami seberapa baik model memahami dan memprediksi sentimen dari nama/merk sepatu. Semakin rendah nilai loss, semakin baik modelnya, karena ini menunjukkan bahwa model dengan lebih akurat memperkirakan label sentimen.

Kedua, accuracy. Akurasi adalah metrik yang mengukur seberapa banyak prediksi model yang benar dari semua prediksi yang dibuat. Dalam kasus ini, akurasi memberi kita gambaran tentang seberapa baik model kita melakukan klasifikasi sentimen secara keseluruhan. Semakin tinggi akurasi, semakin baik modelnya dalam memprediksi sentimen produk sepatu.

Kombinasi kedua metrik ini memberikan pemahaman yang lengkap tentang kinerja model dalam melakukan analisis sentimen terhadap produk sepatu. Loss function memberikan gambaran tentang tingkat presisi model dalam memprediksi sentimen, sementara akurasi memberikan gambaran tentang kinerja keseluruhan model dalam melakukan klasifikasi. Dengan memantau kedua metrik ini, kita dapat memastikan bahwa model yang kita bangun memiliki kinerja yang baik dan dapat diandalkan dalam tugas analisis sentimen.

# Model Selection | Pemilihan Model

Berdasarkan hasil evaluasi, diputuskan untuk memilih model RNN.
"""

# save model (pkl)
with open("sentiment_analysis_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model saved as sentiment_analysis_model.pkl")

"""# Conclusion | Kesimpulan

Berdasarkan eksperimen kita, model RNN memperoleh kinerja yang terbaik, dengan akurasi tes sebesar 0.9796 dan loss tes sebesar 0.1044. Model CNN dan GRU juga memperoleh kinerja yang baik, dengan akurasi yang hampir menyamai model RNN, tetapi nilai loss yang lebih tinggi. Model LSTM, di sisi lain, memperoleh kinerja yang lebih buruk dibandingkan dengan ketiga model lainnya.

Hasil kita menunjukkan bahwa RNN adalah pilihan yang sesuai untuk analisis sentimen dalam ulasan sepatu, dan kinerjanya dapat dikaitkan dengan kemampuan untuk menangkap ketergantungan jangka panjang dalam data teks. Hasil ini memiliki implikasi bagi bisnis dan peneliti yang tertarik dengan analisis sentimen dalam ulasan online.
"""

import pickle

# Save tokenizer
with open("model/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# # Save the best model (assuming it's the RNN model based on your comments)
# model_rnn.save("model/sentiment_analysis_rnn_model.h5")
# print("Tokenizer and model saved successfully.")