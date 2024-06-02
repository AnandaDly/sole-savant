import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import re

# Load dataset
shoe_data = pd.read_csv('../data/Shoes_Data_Final.csv')

# Extract numeric rating from the rating column
shoe_data['numeric_rating'] = shoe_data['rating'].apply(lambda x: float(re.search(r'(\d+\.\d+)', x).group()))

# Select relevant columns
features = shoe_data[['price', 'product_description', 'review_length']]
target = shoe_data['numeric_rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Preprocessing for numeric features
numeric_features = ['price', 'review_length']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for text features
text_features = 'product_description'
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer(max_features=5000))
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('text', text_transformer, text_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'rating_predictor_model.pkl')

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
