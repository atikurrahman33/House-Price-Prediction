import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import os
import requests

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🏠 Real Estate Price Prediction Model Training")
print("=" * 55)

print("1. Loading real housing dataset...")

url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/housing-v8jvYcachJmReGWMXSrnfJep7CJGpG.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Successfully loaded dataset from URL")
    print(f"📊 Dataset shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    try:
        df = pd.read_csv('housing.csv')
        print(f"✅ Loaded from local file")
    except:
        print("❌ Could not load data from URL or local file")
        exit(1)

print("\n2. Data Exploration & Analysis")
print("-" * 35)

print("📈 Dataset Info:")
print(df.info())

print("\n📋 Column Names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print("\n📊 First 5 rows:")
print(df.head())

print("\n🔍 Missing Values:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print(f"⚠️  Total missing values: {missing_values.sum()}")
else:
    print("✅ No missing values found")

print("\n3. Data Cleaning & Preprocessing")
print("-" * 35)

df.columns = df.columns.str.replace(' ', '_').str.replace('.', '').str.replace('(', '').str.replace(')', '')
print("✅ Cleaned column names")

numeric_columns = ['Avg_Area_Income', 'Avg_Area_Number_of_Bedrooms', 'Area_Population']

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
        print(f"✅ Converted {col} to numeric")

df = df.dropna()
print(f"✅ Dataset shape after cleaning: {df.shape}")

print("\n📊 Statistical Summary:")
print(df.describe())

target_col = 'Price'
if target_col not in df.columns:
    print("❌ Price column not found!")
    exit(1)

print(f"\n💰 Price Statistics:")
print(f"Mean Price: ${df[target_col].mean():,.2f}")
print(f"Median Price: ${df[target_col].median():,.2f}")
print(f"Price Range: ${df[target_col].min():,.2f} - ${df[target_col].max():,.2f}")
print(f"Price Std Dev: ${df[target_col].std():,.2f}")

print("\n4. Feature Engineering")
print("-" * 25)

if 'Avg_Area_Income' in df.columns and 'Area_Population' in df.columns:
    df['Income_per_Capita'] = df['Avg_Area_Income'] / df['Area_Population'] * 1000
    print("✅ Created Income_per_Capita feature")

if 'Avg_Area_Number_of_Rooms' in df.columns and 'Avg_Area_Number_of_Bedrooms' in df.columns:
    df['Rooms_per_Bedroom'] = df['Avg_Area_Number_of_Rooms'] / df['Avg_Area_Number_of_Bedrooms']
    print("✅ Created Rooms_per_Bedroom ratio")

if 'Avg_Area_House_Age' in df.columns:
    df['House_Age_Category'] = pd.cut(df['Avg_Area_House_Age'], 
                                     bins=[0, 5, 10, 20, float('inf')], 
                                     labels=['New', 'Recent', 'Mature', 'Old'])
    df['House_Age_Category_encoded'] = df['House_Age_Category'].cat.codes
    print("✅ Created House_Age_Category feature")

if 'Area_Population' in df.columns:
    df['Population_Density_Category'] = pd.cut(df['Area_Population'], 
                                              bins=[0, 20000, 40000, 60000, float('inf')], 
                                              labels=['Low', 'Medium', 'High', 'Very_High'])
    df['Population_Density_encoded'] = df['Population_Density_Category'].cat.codes
    print("✅ Created Population_Density_Category feature")

print("\n5. Correlation Analysis")
print("-" * 25)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Address' in numeric_cols:
    numeric_cols.remove('Address')

correlation_matrix = df[numeric_cols].corr()
price_correlations = correlation_matrix[target_col].sort_values(key=abs, ascending=False)

print("🔗 Features correlation with Price:")
for feature, corr in price_correlations.items():
    if feature != target_col:
        print(f"  {feature}: {corr:.4f}")

print("\n6. Model Building")
print("-" * 20)

feature_columns = [col for col in numeric_cols if col != target_col and col != 'Address']

exclude_cols = ['House_Age_Category', 'Population_Density_Category']
feature_columns = [col for col in feature_columns if col not in exclude_cols]

print(f"📋 Selected features ({len(feature_columns)}):")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")

X = df[feature_columns]
y = df[target_col]

if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("⚠️  Removing remaining NaN values...")
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    print(f"✅ Final dataset shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Linear Regression model trained")

print("\n7. Model Evaluation")
print("-" * 20)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

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

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

print(f"\n🔍 Feature Importance (Top {min(10, len(feature_importance))}):")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['coefficient']:.2f}")

print("\n8. Model Validation")
print("-" * 20)

r2_diff = train_r2 - test_r2
if r2_diff > 0.1:
    print(f"⚠️  Potential overfitting detected (R² difference: {r2_diff:.4f})")
else:
    print(f"✅ Good generalization (R² difference: {r2_diff:.4f})")

residuals = y_test - y_test_pred
residual_std = np.std(residuals)
print(f"📊 Residual standard deviation: ${residual_std:,.2f}")

print("\n9. Saving Model")
print("-" * 15)

os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/house_price_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Model and scaler saved")

model_metadata = {
    'feature_columns': feature_columns,
    'model_performance': {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae)
    },
    'feature_importance': feature_importance.to_dict('records'),
    'data_stats': {
        'n_samples': len(df),
        'n_features': len(feature_columns),
        'price_mean': float(df[target_col].mean()),
        'price_std': float(df[target_col].std()),
        'price_min': float(df[target_col].min()),
        'price_max': float(df[target_col].max())
    },
    'feature_ranges': {
        col: {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        } for col in feature_columns if col in df.columns
    }
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ Model metadata saved")

print("\n10. Sample Predictions")
print("-" * 22)

sample_indices = np.random.choice(X_test.index, 3, replace=False)
print("🔮 Sample predictions vs actual prices:")

for idx in sample_indices:
    sample_features = X_test.loc[idx:idx]
    sample_scaled = scaler.transform(sample_features)
    predicted_price = model.predict(sample_scaled)[0]
    actual_price = y_test.loc[idx]
    error = abs(predicted_price - actual_price)
    error_pct = (error / actual_price) * 100
    
    print(f"\n  Sample {idx}:")
    print(f"    Predicted: ${predicted_price:,.2f}")
    print(f"    Actual:    ${actual_price:,.2f}")
    print(f"    Error:     ${error:,.2f} ({error_pct:.1f}%)")

print(f"\n🎉 Model training complete!")
print(f"📈 Final Test R² Score: {test_r2:.4f}")
print(f"💰 Average prediction error: ${test_mae:,.2f}")

print(f"\n📋 Model Summary:")
print(f"   • Dataset size: {len(df):,} properties")
print(f"   • Features used: {len(feature_columns)}")
print(f"   • Model accuracy: {test_r2:.1%}")
print(f"   • Average error: ${test_mae:,.0f}")
