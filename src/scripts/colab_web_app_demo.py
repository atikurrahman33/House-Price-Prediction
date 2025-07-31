import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np

print("🌐 Interactive Web App Demo for Google Colab")
print("=" * 50)

style = """
<style>
.prediction-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.result-box {
    background: rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
    backdrop-filter: blur(10px);
}
.feature-impact {
    background: #f8f9fa;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border-left: 4px solid #007bff;
}
</style>
"""

display(HTML(style))

income_slider = widgets.IntSlider(
    value=70000,
    min=20000,
    max=200000,
    step=1000,
    description='Area Income ($):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

age_slider = widgets.FloatSlider(
    value=6.0,
    min=0.0,
    max=50.0,
    step=0.1,
    description='House Age (years):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

rooms_slider = widgets.FloatSlider(
    value=7.0,
    min=3.0,
    max=15.0,
    step=0.1,
    description='Avg Rooms:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

bedrooms_slider = widgets.FloatSlider(
    value=4.0,
    min=1.0,
    max=8.0,
    step=0.1,
    description='Avg Bedrooms:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

population_slider = widgets.IntSlider(
    value=35000,
    min=5000,
    max=100000,
    step=500,
    description='Population:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

predict_button = widgets.Button(
    description='🏠 Predict Price',
    button_style='success',
    layout=widgets.Layout(width='200px', height='40px')
)

output_area = widgets.Output()

def predict_and_display(b):
    with output_area:
        clear_output()
        
        try:
            price, confidence = predict_house_price(
                income_slider.value,
                age_slider.value,
                rooms_slider.value,
                bedrooms_slider.value,
                population_slider.value
            )
            
            html_output = f"""
            <div class="prediction-container">
                <h2>🏠 House Price Prediction Results</h2>
                
                <div class="result-box">
                    <h3>💰 Predicted Price: ${price:,.2f}</h3>
                    <p>📊 Confidence Range: ${confidence[0]:,.2f} - ${confidence[1]:,.2f}</p>
                </div>
                
                <div class="result-box">
                    <h4>📋 Your Input Summary:</h4>
                    <p>• Area Income: ${income_slider.value:,}</p>
                    <p>• House Age: {age_slider.value} years</p>
                    <p>• Average Rooms: {rooms_slider.value}</p>
                    <p>• Average Bedrooms: {bedrooms_slider.value}</p>
                    <p>• Area Population: {population_slider.value:,}</p>
                </div>
                
                <div class="result-box">
                    <h4>🎯 Model Performance:</h4>
                    <p>• Accuracy (R²): {test_r2*100:.1f}%</p>
                    <p>• Average Error: ${test_mae:,.0f}</p>
                    <p>• Model Type: Linear Regression</p>
                </div>
            </div>
            """
            
            display(HTML(html_output))
            
        except Exception as e:
            display(HTML(f'<div style="color: red;">Error: {str(e)}</div>'))

predict_button.on_click(predict_and_display)

display(HTML('<h2>🏠 Interactive House Price Predictor</h2>'))
display(HTML('<p>Adjust the sliders below and click "Predict Price" to get instant predictions!</p>'))

display(widgets.VBox([
    income_slider,
    age_slider,
    rooms_slider,
    bedrooms_slider,
    population_slider,
    predict_button,
    output_area
]))

sample_data_html = """
<div style="background: #e3f2fd; padding: 15px; border-radius: 10px; margin: 20px 0;">
    <h3>📊 Quick Test Samples:</h3>
    <p>Try these sample configurations:</p>
    <ul>
        <li><strong>Luxury Area:</strong> Income: $95,000, Age: 3.2 years, Rooms: 8.5, Bedrooms: 5.2, Population: 18,500</li>
        <li><strong>Suburban:</strong> Income: $79,500, Age: 5.7 years, Rooms: 7.0, Bedrooms: 4.1, Population: 23,000</li>
        <li><strong>Urban:</strong> Income: $61,500, Age: 7.8 years, Rooms: 6.2, Bedrooms: 3.6, Population: 40,000</li>
    </ul>
</div>
"""

display(HTML(sample_data_html))
