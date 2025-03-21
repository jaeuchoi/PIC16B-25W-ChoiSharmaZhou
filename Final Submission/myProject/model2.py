import os
from keras import utils
import tensorflow_datasets as tfds
import keras
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf

import os

def parse_labels(label_file, image_root='/content/word_dataset_new_mar20/words'):
    """
    The function parses the label file and extracts valid image paths along with their corresponding labels.

    Args:
        label_file (str): Path to the label file containing image filenames and labels.
        image_root (str, optional): Root directory where images are stored.
                                    

    Returns:
        tuple:
            - image_paths (list of str): A list of valid image file paths.
            - labels (list of str): A list of corresponding labels.
    """

    image_paths = []  # List to store valid image paths
    labels = []  # List to store corresponding labels

    with open(label_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()

            # Validate line format (should have at least 9 fields)
            if len(parts) < 9:
                continue

            # Extract filename and label
            filename = parts[0]  # Example: a01-000u-00-00
            label = parts[-1]  # Label is the last element

            # Construct the path using IAM dataset folder structure:
            # Format: words/{first three chars}/{first seven chars}/{filename}.png
            # Example: words/a01/a01-000u/a01-000u-00-00.png

            folder_level_1 = filename[:3]  # First three characters of filename
            folder_level_2 = "-".join(filename.split("-")[:2])  # Extract first two segments (e.g., a01-000u)
            image_filename = f"{filename}.png"

            image_path = os.path.join(image_root, folder_level_1, folder_level_2, image_filename)

            # Check if the image file exists, add it to lists if valid
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(label)

    return image_paths, labels



def encode_label(label):
    """
    This function encodes a text label into a numerical representation.

    Args:
        label (str): The text label to be encoded.

    Returns:
        tf.Tensor: A tensor containing the numerical encoding of the label.
        ```
    """

    # Split the label into individual characters 
    label_chars = tf.strings.unicode_split(label, input_encoding='UTF-8')

    # Convert characters into numerical encoding using a predefined mapping
    label_encoded = char_to_num(label_chars)

    return label_encoded


def load_and_preprocess_image(image_path):
    """
    The function loads an image from a file path and applies preprocessing.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: A preprocessed image tensor with shape.
    """

    # Read the image file as raw bytes
    img = tf.io.read_file(image_path)

    # Decode the image as a grayscale PNG
    img = tf.image.decode_png(img, channels=1)

    # Resize the image to a fixed height and width
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])

    # Normalize pixel values to [0, 1] range
    img = img / 255.0

    return img


def preprocess_dataset(image_path, label):
    """
    This function preprocesses a dataset sample by applying image preprocessing and label encoding.

    Args:
        image_path (str): Path to the image file.
        label (str): Text label corresponding to the image.

    Returns:
        tuple: A dictionary containing the processed image and its encoded label.
            - dict: {"image": tf.Tensor} - Preprocessed image tensor.
            - tf.Tensor: Encoded label tensor.
    """

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path)

    # Encode the label into numerical representation
    label = encode_label(label)

    return ({"image": image}, label)


def ctc_loss(y_true, y_pred):
    """
    This function computes the Connectionist Temporal Classification (CTC) loss for sequence modeling.

    Args:
        y_true (tf.Tensor): Ground truth labels (batch_size, label_sequence_length).
        y_pred (tf.Tensor): Model predictions (batch_size, time_steps, num_classes).

    Returns:
        tf.Tensor: Computed CTC loss value for the batch.

    Explanation:
    - `batch_len`: Determines the batch size.
    - `input_length`: Computes the input sequence lengths for each batch element.
    - `label_length`: Computes the label sequence lengths for each batch element.
    - `ctc_batch_cost`: Keras backend function to compute CTC loss.
        ```
    """

    # Compute batch size
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")

    # Compute the input length for CTC loss calculation
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64") * tf.ones(shape=(batch_len, 1), dtype="int64")

    # Compute the label length for CTC loss calculation
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64") * tf.ones(shape=(batch_len, 1), dtype="int64")

    # Compute CTC loss using Keras backend function
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    return loss


def decode_ctc(predicted_indices):
    """
    This function decodes the predicted indices from a CTC model into a readable text format.

    Args:
        predicted_indices (tf.Tensor): A tensor containing predicted character indices.

    Returns:
        str: The decoded text string.
    """

    # Convert numerical indices to characters
    chars = num_to_char(predicted_indices)

    # Convert tensor to a string by joining characters along the last axis
    decoded_text = tf.strings.reduce_join(chars, axis=-1)

    # Convert to a NumPy array and decode as UTF-8 string
    decoded_text = decoded_text.numpy()[0].decode('utf-8')

    # Remove '[UNK]' tokens and blank characters while preventing duplicate letters
    final_text = []
    prev_char = ''

    for char in decoded_text:
        if char not in ['[UNK]', '', ' '] and char != prev_char:  # Ignore blanks and duplicate characters
            final_text.append(char)
            prev_char = char  # Track last appended character

    return ''.join(final_text)


def preprocess_single_image(image_path):
    """
    This function preprocesses a single image for input into a neural network.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: A preprocessed image tensor of shape (1, IMAGE_HEIGHT, IMAGE_WIDTH, 1).
    """

    # Read the image file as raw bytes
    img = tf.io.read_file(image_path)

    # Decode the image as a grayscale PNG
    img = tf.image.decode_png(img, channels=1)

    # Resize the image to a fixed height and width
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Expand dimensions to add batch size of 1 
    img = tf.expand_dims(img, axis=0)

    return img


def predict_text(model, image_path):
    """
    This function predicts the text from an image using a trained model.

    Args:
        model (tf.keras.Model): The trained model for text recognition.
        image_path (str): Path to the input image file.

    Returns:
        str: The predicted text string.
    """

    # Preprocess the image for model input
    img_tensor = preprocess_single_image(image_path)

    # Generate predictions using the trained model
    prediction = model.predict(img_tensor)

    # Convert prediction logits into character indices
    predicted_indices = tf.argmax(prediction, axis=-1)

    # Debugging output to check predicted indices
    print("Predicted indices:", predicted_indices.numpy())

    # Decode predicted indices into human-readable text
    return decode_ctc(predicted_indices)


def preprocess_image(contents):
    """
    This function preprocesses an image received as a base64-encoded string.

    Args:
        contents (str): Base64-encoded image string.

    Returns:
        np.ndarray: A preprocessed image tensor of shape.

    """

    # Extract base64-encoded image data 
    image_data = contents.split(",")[1]

    # Decode base64 string into raw image bytes
    img_bytes = base64.b64decode(image_data)

    # Open image using PIL and convert to grayscale
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize the image to the required input dimensions
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert image to a NumPy array and normalize pixel values to [0, 1]
    img_array = np.array(img).astype("float32") / 255.0

    # Expand dimensions to match model input shape: (batch_size=1, height, width, channels=1)
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array

def predict_text1(image_tensor):
    """
    This function predicts the text from a preprocessed image tensor using the trained model.

    Args:
        image_tensor (tf.Tensor): Preprocessed image tensor of shape (1, IMAGE_HEIGHT, IMAGE_WIDTH, 1).

    Returns:
        str: The cleaned predicted text string.
    """

    # Generate predictions using the trained model
    prediction = model.predict(image_tensor)

    # Convert predicted logits into character indices
    predicted_indices = tf.argmax(prediction, axis=-1)

    # Decode predicted indices into text and remove unknown tokens
    cleaned_text = decode_ctc(predicted_indices).replace("[UNK]", " ")

    return cleaned_text


def find_free_port():
    """
    This function finds and returns an available network port on the local machine.

    Returns:
        int: A free port number that can be used for network communication.
    """

    # Create a socket object 
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to an available port 
        s.bind(('', 0))

        # Retrieve and return the assigned port number
        return s.getsockname()[1]


def handle_image(contents):
    """
    The function handles an uploaded image, processes it, and returns a styled HTML component displaying the image and detected text.

    Args:
        contents (str or None): Base64-encoded image string. If None, the function returns an empty string.

    Returns:
        html.Div or str: A Dash HTML component containing:
            - The uploaded image.
            - The detected text styled with a green glow effect.
        Returns an empty string if no image is provided.
    """

    if contents is not None:
        # Preprocess the base64-encoded image to a model-compatible tensor
        image_tensor = preprocess_image(contents)

        # Perform text prediction using the trained model
        predicted_word = predict_text1(image_tensor)

        # Return the formatted HTML component displaying the image and detected text
        return html.Div([
            # Display the uploaded image
            html.Img(src=contents, style={
                'maxWidth': '400px',  # Limit maximum width
                'borderRadius': '5px',  # Apply rounded corners
                'marginBottom': '20px',  # Add spacing below the image
                'boxShadow': '0 0 20px #00FF00'  # Green glow effect
            }),

            # Display the "DETECTED:" header
            html.H3("DETECTED:", style={
                'color': '#00FF00', 
                'textShadow': '0 0 5px #00FF00' 
            }),

            # Display the predicted word
            html.Div(predicted_word, style={
                'fontSize': '32px',  
                'fontWeight': 'bold',  
                'color': '#00FF00', 
                'textShadow': '0 0 8px #00FF00'  
            })
        ])

    return ""


