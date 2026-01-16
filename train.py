import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

# Load data
print("✓ Loading data...")
resumes_df = pd.read_csv('data/resumes_clean.csv')
labels_df = pd.read_csv('data/labels_clean.csv')

print(f"✓ Loaded {len(resumes_df)} resumes")

# Prepare data
X = resumes_df['resume_text'].values
y = labels_df['score'].values

# Split data: 80% training, 20% testing
print("✓ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text (TF-IDF)
print("✓ Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("✓ Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_vec, y_train)

# Evaluate
print("✓ Evaluating model...")
y_pred = model.predict(X_test_vec)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = (1 - (mse / (max(y)**2))) * 100

print(f"\n--- MODEL PERFORMANCE ---")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Estimated Accuracy: {accuracy:.2f}%")

# Save model and vectorizer
print("\n✓ Saving model...")
os.makedirs('models', exist_ok=True)
pickle.dump(model, open('models/trained_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

print("✅ Successfully trained and saved model!")
print("✅ Models saved in: models/")
print(f"\nYou can now run: streamlit run app.py")
