import dash
from dash import html, dcc, Output, Input
import base64
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import socket

# Load model
model = tf.keras.models.load_model("ocr_model_8.keras", compile=False)

# Image dimensions from training
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 300

# Character decoding function
characters = ['!', '"', "'", '(', ')', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, num_oov_indices=1)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), num_oov_indices=1, invert=True, mask_token=None)

def preprocess_image(contents):
    image_data = contents.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

def decode_ctc(predicted_indices):
    chars = num_to_char(predicted_indices)
    decoded_text = tf.strings.reduce_join(chars, axis=-1)
    decoded_text = decoded_text.numpy()[0].decode("utf-8")
    final_text = []
    prev_char = ''
    for char in decoded_text:
        if char not in ['[UNK]', '', ' '] and char != prev_char:
            final_text.append(char)
            prev_char = char
    return ''.join(final_text)

def predict_text(image_tensor):
    prediction = model.predict(image_tensor)
    predicted_indices = tf.argmax(prediction, axis=-1)
    cleaned_text = decode_ctc(predicted_indices).replace("[UNK]", " ")
    return cleaned_text

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# Build the Dash app
app = dash.Dash(__name__)
app.title = "OCR Console"

app.layout = html.Div([
    html.H1("OCR - Image Word Recognizer", style={
        'textAlign': 'center',
        'color': '#00FF00',
        'fontFamily': 'Courier New',
        'fontSize': '36px',
        'marginBottom': '30px',
        'textShadow': '0 0 10px #00FF00'
    }),

    dcc.Upload(
        id='upload-image',
        children=html.Div(['[ DROP FILE HERE ] or ', html.A('SELECT IMAGE')]),
        style={
            'width': '70%',
            'height': '120px',
            'lineHeight': '120px',
            'borderWidth': '2px',
            'borderStyle': 'dashed',
            'borderRadius': '10px',
            'textAlign': 'center',
            'margin': 'auto',
            'backgroundColor': '#000000',
            'color': '#00FF00',
            'fontSize': '20px',
            'fontFamily': 'Courier New',
            'transition': '0.3s ease',
            'textShadow': '0 0 5px #00FF00'
        },
        multiple=False
    ),

    html.Div(id='output', style={
        'textAlign': 'center',
        'marginTop': '40px',
        'color': '#00FF00',
        'fontFamily': 'Courier New, monospace',
        'fontSize': '20px'
    }),
], style={
    'backgroundColor': '#000000',
    'height': '100vh',
    'padding': '40px'
})

@app.callback(
    Output('output', 'children'),
    Input('upload-image', 'contents')
)
def handle_image(contents):
    if contents is not None:
        image_tensor = preprocess_image(contents)
        predicted_word = predict_text(image_tensor)
        return html.Div([
            html.Img(src=contents, style={
                'maxWidth': '400px',
                'borderRadius': '5px',
                'marginBottom': '20px',
                'boxShadow': '0 0 20px #00FF00'
            }),
            html.H3("DETECTED:", style={
                'color': '#00FF00',
                'textShadow': '0 0 5px #00FF00'
            }),
            html.Div(predicted_word, style={
                'fontSize': '32px',
                'fontWeight': 'bold',
                'color': '#00FF00',
                'textShadow': '0 0 8px #00FF00'
            })
        ])
    return ""

if __name__ == '__main__':
    port = find_free_port()
    print(f"Running on http://127.0.0.1:{port}")
    app.run_server(debug=True, port=port)