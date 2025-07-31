

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
import requests
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

print("🏠 Real Estate Price Prediction - Google Colab Version")
print("=" * 60)

print("1. Loading housing dataset...")

url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/housing-v8jvYcachJmReGWMXSrnfJep7CJGpG.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Successfully loaded dataset from URL")
    print(f"📊 Dataset shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("Please upload your housing.csv file manually")
    uploaded = files.upload()
    df = pd.read_csv(list(uploaded.keys())[0])
    print(f"✅ Loaded from uploaded file")

print("\n2. Data Exploration & Analysis")
print("-" * 40)

print("📈 Dataset Info:")
print(df.info())

print("\n📋 Column Names:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

print("\n📊 First 5 rows:")
display(df.head())

print("\n🔍 Missing Values:")
missing_values = df.isnull().sum()
print(missing_values)

if missing_values.sum() > 0:
    print(f"⚠️  Total missing values: {missing_values.sum()}")
else:
    print("✅ No missing values found")

print("\n3. Data Cleaning & Preprocessing")
print("-" * 40)

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
display(df.describe())

target_col = 'Price'
if target_col not in df.columns:
    print("❌ Price column not found!")
    exit(1)

print(f"\n💰 Price Statistics:")
print(f"Mean Price: ${df[target_col].mean():,.2f}")
print(f"Median Price: ${df[target_col].median():,.2f}")
print(f"Price Range: ${df[target_col].min():,.2f} - ${df[target_col].max():,.2f}")
print(f"Price Std Dev: ${df[target_col].std():,.2f}")

print("\n4. Data Visualization")
print("-" * 25)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Housing Data Exploration', fontsize=16, fontweight='bold')

axes[0, 0].hist(df['Price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Price Distribution')
axes[0, 0].set_xlabel('Price ($)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].scatter(df['Avg_Area_Income'], df['Price'], alpha=0.6, color='green')
axes[0, 1].set_title('Price vs Area Income')
axes[0, 1].set_xlabel('Average Area Income ($)')
axes[0, 1].set_ylabel('Price ($)')

axes[0, 2].scatter(df['Avg_Area_Number_of_Rooms'], df['Price'], alpha=0.6, color='orange')
axes[0, 2].set_title('Price vs Number of Rooms')
axes[0, 2].set_xlabel('Average Number of Rooms')
axes[0, 2].set_ylabel('Price ($)')

axes[1, 0].scatter(df['Avg_Area_House_Age'], df['Price'], alpha=0.6, color='red')
axes[1, 0].set_title('Price vs House Age')
axes[1, 0].set_xlabel('Average House Age (years)')
axes[1, 0].set_ylabel('Price ($)')

axes[1, 1].scatter(df['Area_Population'], df['Price'], alpha=0.6, color='purple')
axes[1, 1].set_title('Price vs Area Population')
axes[1, 1].set_xlabel('Area Population')
axes[1, 1].set_ylabel('Price ($)')

axes[1, 2].scatter(df['Avg_Area_Number_of_Bedrooms'], df['Price'], alpha=0.6, color='brown')
axes[1, 2].set_title('Price vs Number of Bedrooms')
axes[1, 2].set_xlabel('Average Number of Bedrooms')
axes[1, 2].set_ylabel('Price ($)')

plt.tight_layout()
plt.show()

print("\n5. Feature Engineering")
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

print("\n6. Correlation Analysis")
print("-" * 30)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Address' in numeric_cols:
    numeric_cols.remove('Address')

correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

price_correlations = correlation_matrix[target_col].sort_values(key=abs, ascending=False)

print("🔗 Features correlation with Price:")
for feature, corr in price_correlations.items():
    if feature != target_col:
        print(f"  {feature}: {corr:.4f}")

print("\n7. Model Building")
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

print("\n8. Model Evaluation")
print("-" * 25)

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
display(feature_importance.head(10))

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
colors = ['green' if x > 0 else 'red' for x in top_features['coefficient']]
plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Coefficient Value')
plt.title('Top 10 Feature Importance (Linear Regression Coefficients)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n9. Model Validation & Visualization")
print("-" * 35)

r2_diff = train_r2 - test_r2
if r2_diff > 0.1:
    print(f"⚠️  Potential overfitting detected (R² difference: {r2_diff:.4f})")
else:
    print(f"✅ Good generalization (R² difference: {r2_diff:.4f})")

residuals = y_test - y_test_pred
residual_std = np.std(residuals)
print(f"📊 Residual standard deviation: ${residual_std:,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')

axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title(f'Actual vs Predicted (R² = {test_r2:.3f})')

axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, color='green')
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted Price ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot')

axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].set_xlabel('Residuals ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Residual Distribution')

performance_metrics = ['R²', 'RMSE', 'MAE']
train_values = [train_r2, train_rmse/1000, train_mae/1000]
test_values = [test_r2, test_rmse/1000, test_mae/1000]

x = np.arange(len(performance_metrics))
width = 0.35

axes[1, 1].bar(x - width/2, train_values, width, label='Training', alpha=0.7, color='skyblue')
axes[1, 1].bar(x + width/2, test_values, width, label='Testing', alpha=0.7, color='lightcoral')
axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Values (RMSE & MAE in $1000s)')
axes[1, 1].set_title('Training vs Testing Performance')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(performance_metrics)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n10. Sample Predictions")
print("-" * 25)

sample_indices = np.random.choice(X_test.index, 5, replace=False)
print("🔮 Sample predictions vs actual prices:")

predictions_df = pd.DataFrame()
for i, idx in enumerate(sample_indices):
    sample_features = X_test.loc[idx:idx]
    sample_scaled = scaler.transform(sample_features)
    predicted_price = model.predict(sample_scaled)[0]
    actual_price = y_test.loc[idx]
    error = abs(predicted_price - actual_price)
    error_pct = (error / actual_price) * 100
    
    predictions_df = pd.concat([predictions_df, pd.DataFrame({
        'Sample': [f'Sample {i+1}'],
        'Predicted': [f'${predicted_price:,.2f}'],
        'Actual': [f'${actual_price:,.2f}'],
        'Error': [f'${error:,.2f}'],
        'Error %': [f'{error_pct:.1f}%']
    })], ignore_index=True)

display(predictions_df)

print("\n11. Interactive Prediction Function")
print("-" * 35)

def predict_house_price(avg_area_income, avg_area_house_age, avg_area_number_of_rooms, 
                       avg_area_number_of_bedrooms, area_population):
    
    input_data = pd.DataFrame({
        'Avg_Area_Income': [avg_area_income],
        'Avg_Area_House_Age': [avg_area_house_age],
        'Avg_Area_Number_of_Rooms': [avg_area_number_of_rooms],
        'Avg_Area_Number_of_Bedrooms': [avg_area_number_of_bedrooms],
        'Area_Population': [area_population]
    })
    
    input_data['Income_per_Capita'] = input_data['Avg_Area_Income'] / input_data['Area_Population'] * 1000
    input_data['Rooms_per_Bedroom'] = input_data['Avg_Area_Number_of_Rooms'] / input_data['Avg_Area_Number_of_Bedrooms']
    
    house_age_category_encoded = 0
    if avg_area_house_age > 20: house_age_category_encoded = 3
    elif avg_area_house_age > 10: house_age_category_encoded = 2
    elif avg_area_house_age > 5: house_age_category_encoded = 1
    input_data['House_Age_Category_encoded'] = house_age_category_encoded
    
    population_density_encoded = 0
    if area_population > 60000: population_density_encoded = 3
    elif area_population > 40000: population_density_encoded = 2
    elif area_population > 20000: population_density_encoded = 1
    input_data['Population_Density_encoded'] = population_density_encoded
    
    input_scaled = scaler.transform(input_data[feature_columns])
    predicted_price = model.predict(input_scaled)[0]
    
    confidence_range = predicted_price * 0.12
    confidence_interval = [predicted_price - confidence_range, predicted_price + confidence_range]
    
    print(f"🏠 Predicted House Price: ${predicted_price:,.2f}")
    print(f"📊 Confidence Range: ${confidence_interval[0]:,.2f} - ${confidence_interval[1]:,.2f}")
    
    return predicted_price, confidence_interval

print("✅ Prediction function created!")
print("\nExample usage:")
print("predict_house_price(75000, 6.5, 7.2, 4.1, 35000)")

example_price, example_range = predict_house_price(75000, 6.5, 7.2, 4.1, 35000)

print("\n12. Model Export")
print("-" * 20)

model_data = {
    'model': model,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_performance': {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae)
    },
    'feature_importance': feature_importance.to_dict('records')
}

joblib.dump(model_data, 'house_price_model_complete.pkl')
print("✅ Complete model saved as 'house_price_model_complete.pkl'")

files.download('house_price_model_complete.pkl')
print("📥 Model file downloaded to your computer")

print(f"\n🎉 Analysis Complete!")
print("=" * 50)
print(f"📈 Final Model Performance:")
print(f"   • Test R² Score: {test_r2:.4f} ({test_r2*100:.1f}% accuracy)")
print(f"   • Average Prediction Error: ${test_mae:,.2f}")
print(f"   • Dataset Size: {len(df):,} properties")
print(f"   • Features Used: {len(feature_columns)}")
print(f"   • Model Type: Linear Regression with Feature Engineering")

print(f"\n📋 Key Insights:")
print(f"   • Most Important Feature: {feature_importance.iloc[0]['feature']}")
print(f"   • Price Range: ${df[target_col].min():,.0f} - ${df[target_col].max():,.0f}")
print(f"   • Average Price: ${df[target_col].mean():,.0f}")
print(f"   • Model Generalization: {'Good' if r2_diff <= 0.1 else 'Needs Improvement'}")
