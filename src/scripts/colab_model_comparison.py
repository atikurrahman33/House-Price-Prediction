from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import time

print("🔬 Model Comparison Analysis")
print("=" * 40)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Support Vector Regression': SVR(kernel='rbf', C=1000, gamma=0.1)
}

results = {}

print("Training and evaluating models...")

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    start_time = time.time()
    
    if name == 'Support Vector Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Training Time': training_time,
        'Predictions': y_pred
    }
    
    print(f"✅ {name} completed in {training_time:.2f}s")
    print(f"   R²: {r2:.4f}, RMSE: ${rmse:,.0f}, MAE: ${mae:,.0f}")

results_df = pd.DataFrame({
    name: {
        'R² Score': results[name]['R²'],
        'RMSE ($)': results[name]['RMSE'],
        'MAE ($)': results[name]['MAE'],
        'Training Time (s)': results[name]['Training Time']
    }
    for name in results.keys()
}).T

print("\n📊 Model Comparison Results:")
display(results_df.round(4))

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['R² Score', 'RMSE ($)', 'MAE ($)', 'Training Time (s)']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    values = results_df[metric].values
    bars = ax.bar(results_df.index, values, color=colors[i], alpha=0.7, edgecolor='black')
    ax.set_title(f'{metric} Comparison')
    ax.set_ylabel(metric)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metric == 'R² Score':
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        elif 'Time' in metric:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.2f}s', ha='center', va='bottom')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                   f'${value:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

best_model = results_df['R² Score'].idxmax()
print(f"\n🏆 Best Performing Model: {best_model}")
print(f"   R² Score: {results_df.loc[best_model, 'R² Score']:.4f}")
print(f"   RMSE: ${results_df.loc[best_model, 'RMSE ($)']:,.0f}")
print(f"   MAE: ${results_df.loc[best_model, 'MAE ($)']:,.0f}")

plt.figure(figsize=(12, 8))
for name in results.keys():
    plt.scatter(y_test, results[name]['Predictions'], alpha=0.6, label=name, s=30)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Actual vs Predicted Prices - All Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
