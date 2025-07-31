import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🏠 House Price Prediction Model Training")
print("=" * 50)

# 1. DATA GENERATION (Since housing.csv wasn't provided)
print("1. Generating sample housing dataset...")

np.random.seed(42)
n_samples = 1000

# Generate synthetic housing data
data = {
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.randint(800, 4000, n_samples),
    'sqft_lot': np.random.randint(2000, 15000, n_samples),
    'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples),
    'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(3, 13, n_samples),
    'yr_built': np.random.randint(1900, 2023, n_samples),
    'yr_renovated': np.random.choice([0] + list(range(1950, 2023)), n_samples, p=[0.7] + [0.3/73]*73),
    'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005', '98006'], n_samples),
    'lat': np.random.uniform(47.1, 47.8, n_samples),
    'long': np.random.uniform(-122.5, -121.3, n_samples)
}

df = pd.DataFrame(data)

# Create realistic price based on features
price_base = (
    df['sqft_living'] * 150 +
    df['bedrooms'] * 10000 +
    df['bathrooms'] * 15000 +
    df['waterfront'] * 200000 +
    df['view'] * 20000 +
    df['condition'] * 10000 +
    df['grade'] * 25000 +
    (2023 - df['yr_built']) * -500 +
    np.where(df['yr_renovated'] > 0, 50000, 0)
)

# Add some noise and ensure positive prices
df['price'] = np.maximum(price_base + np.random.normal(0, 50000, n_samples), 100000)

print(f"✅ Generated {len(df)} housing records")
print(f"📊 Dataset shape: {df.shape}")

# 2. DATA EXPLORATION
print("\n2. Data Exploration & Analysis")
print("-" * 30)

print("📈 Dataset Info:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe())

print("\n🔍 Missing Values:")
print(df.isnull().sum())

print("\n💰 Price Statistics:")
print(f"Mean Price: ${df['price'].mean():,.2f}")
print(f"Median Price: ${df['price'].median():,.2f}")
print(f"Price Range: ${df['price'].min():,.2f} - ${df['price'].max():,.2f}")

# 3. DATA CLEANING & FEATURE ENGINEERING
print("\n3. Feature Engineering")
print("-" * 25)

# Create new features
df['house_age'] = 2023 - df['yr_built']
df['renovated'] = (df['yr_renovated'] > 0).astype(int)
df['years_since_renovation'] = np.where(df['yr_renovated'] > 0, 2023 - df['yr_renovated'], 0)
df['price_per_sqft'] = df['price'] / df['sqft_living']
df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# Encode categorical variables
le_zipcode = LabelEncoder()
df['zipcode_encoded'] = le_zipcode.fit_transform(df['zipcode'])

print("✅ Created new features:")
print("   - house_age, renovated, years_since_renovation")
print("   - price_per_sqft, bed_bath_ratio, total_rooms")
print("   - zipcode_encoded")

# 4. MODEL BUILDING
print("\n4. Model Building")
print("-" * 20)

# Select features for modeling
feature_columns = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'house_age',
    'renovated', 'years_since_renovation', 'zipcode_encoded',
    'lat', 'long', 'bed_bath_ratio', 'total_rooms'
]

X = df[feature_columns]
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✅ Linear Regression model trained")

# 5. MODEL EVALUATION
print("\n5. Model Evaluation")
print("-" * 20)

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Calculate metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("📊 Model Performance:")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Training RMSE: ${train_rmse:,.2f}")
print(f"Testing RMSE: ${test_rmse:,.2f}")
print(f"Training MAE: ${train_mae:,.2f}")
print(f"Testing MAE: ${test_mae:,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print("\n🔍 Top 10 Most Important Features:")
print(feature_importance.head(10)[['feature', 'coefficient']])

# 6. SAVE MODEL AND METADATA
print("\n6. Saving Model")
print("-" * 15)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model and scaler
joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le_zipcode, 'models/zipcode_encoder.pkl')

# Save feature columns and model metadata
model_metadata = {
    'feature_columns': feature_columns,
    'model_performance': {
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae)
    },
    'feature_importance': feature_importance.to_dict('records'),
    'zipcode_classes': le_zipcode.classes_.tolist()
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ Model saved successfully!")
print("   - house_price_model.pkl")
print("   - scaler.pkl") 
print("   - zipcode_encoder.pkl")
print("   - model_metadata.json")

print(f"\n🎉 Model training complete! Test R² Score: {test_r2:.4f}")
