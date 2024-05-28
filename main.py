import streamlit as st

from multiapp import MultiApp

from apps import home, login, register
import app as main_app

multi_app = MultiApp()

st.markdown("""
# Sole Savant

""")

# Tambahkan semua aplikasi Anda di sini
multi_app.add_app("Home", home.app)
multi_app.add_app("Login", login.app)
multi_app.add_app("Register", register.app)
multi_app.add_app("App", main_app.app)  # Menggunakan main_app.app

# Menjalankan aplikasi utama
multi_app.run()