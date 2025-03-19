#!/usr/bin/env python
# coding: utf-8

# In[8]:


import dash
from dash import dcc, html, Input, Output
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

# Load the trained OCR model
MODEL_PATH = "/Users/jaeuchoi/Downloads/ocr_model_4.keras"  # Update with actual path
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Ensure `char_to_num` and `num_to_char` are initialized correctly
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
char_to_num = tf.keras.layers.StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), invert=True, mask_token=None)

# Image dimensions used in training
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 300  # Update if different

# Dash app setup
app = dash.Dash(__name__)
app.title = "Cozy OCR Web App"

# Function to process uploaded image
def preprocess_single_image(image_data):
    """Processes a user-uploaded image into a format for the OCR model."""
    image = Image.open(io.BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=[0, -1])  # Add batch & channel dims
    return image

# Function to decode CTC predictions
def decode_ctc(predicted_indices):
    chars = num_to_char(predicted_indices)
    decoded_text = tf.strings.reduce_join(chars, axis=-1).numpy()[0].decode('utf-8')

    # Remove all occurrences of '[UNK]'
    cleaned_text = decoded_text.replace("[UNK]", " ")

    return cleaned_text

# Dash layout
app.layout = html.Div([
    html.H1("Handwritten OCR Model", className="title"),

    dcc.Upload(
        id="upload-image",
        children=html.Button("Upload an Image", className="upload-btn"),
        multiple=False
    ),

    html.Div(id="output-container", className="output-container"),

    html.Img(id="uploaded-image", className="uploaded-image"),

    html.H3("Recognized Text:", className="recognized-text-title"),
    html.Div(id="output-text", className="recognized-text")
], className="container")

# Callback to handle image upload
@app.callback(
    [Output("uploaded-image", "src"),
     Output("output-text", "children")],
    [Input("upload-image", "contents")]
)
def update_output(image_contents):
    if image_contents is None:
        return None, "Upload an image to get started."

    # Decode image from base64
    content_type, content_string = image_contents.split(',')
    decoded = base64.b64decode(content_string)

    # Preprocess image
    processed_img = preprocess_single_image(decoded)

    # Get model predictions
    prediction = model.predict(processed_img)
    predicted_indices = tf.argmax(prediction, axis=-1)
    detected_text = decode_ctc(predicted_indices)  # Uses updated function

    # Convert image back to base64 for display
    encoded_image = f"data:image/png;base64,{content_string}"

    return encoded_image, f"Detected Word: {detected_text}"



# Add custom CSS styles for cozy look
app.css.append_css({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
})

app.index_string = '''
<!DOCTYPE html>
<html>
<head>
    <title>Cozy OCR Web App</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #FAF3E0; /* Warm beige background */
            text-align: center;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: auto;
            background: #fff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .title {
            font-size: 32px;
            color: #4A3F35;
        }
        .upload-btn {
            background-color: #FFAD60;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }
        .upload-btn:hover {
            background-color: #FF8C42;
            transform: scale(1.1);
        }
        .uploaded-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            margin-top: 20px;
            opacity: 0;
            animation: fadeIn 1s ease-in-out forwards;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .recognized-text-title {
            font-size: 22px;
            color: #4A3F35;
            margin-top: 20px;
        }
        .recognized-text {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            background: #FDEBD0;
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease-in-out;
        }
        .recognized-text:hover {
            transform: scale(1.05);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">PIC 16B OCR Project</h1>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </div>
</body>
</html>
'''



import socket

def find_available_port():
    """Finds an available port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to any available port
        return s.getsockname()[1]  # Get the assigned port

# Run the Dash app on an available port
if __name__ == "__main__":
    port = find_available_port()
    print(f"Running Dash app on port {port}")
    app.run_server(debug=True, host="0.0.0.0", port=port)


# In[ ]:




