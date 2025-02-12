import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

# Load trained models

m = pickle.load(open('logistic_regression_model.pkl', 'rb')) # Logistic Regression
nn = pickle.load(open('neural_network_model.pkl', 'rb')) # Neural Network
RFC = pickle.load(open('random_forest_model.pkl', 'rb')) # Random Forest Classifier
 
# Image Processing Function
def image_to_8x8_array(image_file):
    img = Image.open(image_file).convert("L")  # Convert to grayscale
    img = img.resize((8, 8), resample=Image.LANCZOS)  # Resize to 8x8
    arr = np.array(img)  # Convert to NumPy array
    arr_inverted = 255 - arr  # Invert colors (black → white)
    arr_8x8 = arr_inverted // 16  # Scale to range [0,16]
    return arr_8x8

# Dash App Setup
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Handwritten Digit Recognizer"),
    
    dcc.Upload(
        id="upload-image",
        children=html.Button("Upload Image", style={'fontSize': 18}),
        multiple=False
    ),
    
    html.Div(id="output-prediction"),
    
    html.Div(id="output-image"),
])

@app.callback(
    [Output("output-image", "children"), Output("output-prediction", "children")],
    [Input("upload-image", "contents")],
    [State("upload-image", "filename")]
)
def update_output(contents, filename):
    if contents is None:
        return None, None

    # Decode image from base64
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image_file = io.BytesIO(decoded)
    
    # Convert image to 8x8 array
    img_array = image_to_8x8_array(image_file)
    
    # Flatten image array to 1D (matching trained model input)
    digit_data = img_array.flatten().reshape(1, -1)
    
    # Predictions
    prediction_m = m.predict(digit_data)[0]
    prediction_nn = nn.predict(digit_data)[0]
    prediction_rfc = RFC.predict(digit_data)[0]

    # Plot the 8x8 processed image
    # px.imshow with color_continuous_scale = gray use 16 as white and 0 as black. Since our data uses reverse scale, reverse for accurate depiction
    imgarr_for_px = 16 - img_array 
    fig = px.imshow(imgarr_for_px, color_continuous_scale="gray", title="Processed Image (8x8)")
    
    
    
    # Display predictions and processed image
    prediction_text = f"""
    {img_array}
    Predictions:
    1. Logistic Regression: {prediction_m}
    2. Neural Network: {prediction_nn}
    3. Random Forest: {prediction_rfc}
    """

    return dcc.Graph(figure=fig), html.Pre(prediction_text)

# Run App
if __name__ == "__main__":
    app.run_server(debug=True)
