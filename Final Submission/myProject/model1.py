import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import string
import random
import editdistance
import orbax.checkpoint
from flax import linen as nn
from PIL import Image, UnidentifiedImageError

class IAMDataset:
    """
    A dataset class for the IAM Handwriting Database.

    This class loads image-label pairs from a given directory and label file.
    It processes text labels into numerical form and ensures valid image paths.
    """

    def __init__(self, img_root, label_file, img_size=(32, 128)):
        """
        Initializes the IAMDataset instance.

        Args:
            img_root (str): Root directory where images are stored.
            label_file (str): Path to the label file containing text annotations.
            img_size (tuple, optional): Size to which images should be resized. 
        """
        self.img_root = img_root
        self.img_size = img_size
        self.valid_samples = []  # Stores valid image-label pairs

        # Read label file and process each entry
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):  # Ignore comments 
                    continue

                # Extract relevant parts of the label file
                parts = line.strip().split(" ")
                img_name = parts[0]  # Image filename
                text = " ".join(parts[8:]).lower()  # Extract the transcription 

                # Convert text to numerical labels using a predefined mapping
                label = [char_to_index[c] for c in text if c in char_to_index]

                # Construct the full image path based on IAM dataset folder structure
                folder1, folder2 = img_name.split("-")[:2]
                img_path = os.path.normpath(os.path.join(self.img_root, folder1, f"{folder1}-{folder2}", img_name + ".png"))

                # Check if the image exists and is valid, then add to dataset
                if os.path.exists(img_path) and self._is_valid_image(img_path):
                    self.valid_samples.append((img_path, label))

    def __len__(self):
        """
        Returns the number of valid samples in the dataset.

        Returns:
            int: Total number of valid image-label pairs.
        """
        return len(self.valid_samples)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: Processed grayscale image as a NumPy array.
                - jnp.ndarray: Corresponding numerical label array.
        """
        img_path, label = self.valid_samples[idx]

        try:
            # Open image in grayscale mode
            image = Image.open(img_path).convert("L")

            # Resize the image to a fixed size
            image = image.resize((128, 32), Image.BILINEAR)

            # Convert image to a NumPy array and normalize pixel values to [0, 1]
            image = np.array(image, dtype=np.float32) / 255.0

            # Add a channel dimension (height, width, 1) for compatibility with models
            image = np.expand_dims(image, axis=-1)

            return image, jnp.array(label, dtype=jnp.int32)

        except (UnidentifiedImageError, OSError):
            return None  # Return None if the image file is corrupted 

    def _is_valid_image(self, img_path):
        """
        Checks if an image file is valid and not corrupted.

        Args:
            img_path (str): Path to the image file.

        Returns:
            bool: True if the image file is valid, False otherwise.
        """
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify that the image is not corrupted
            return True
        except (UnidentifiedImageError, OSError):
            return False  # Return False if image verification fails


def jax_dataloader(dataset, batch_size=32, shuffle=True):
    """
    A data loader function for JAX that generates batches from a dataset.

    This function iterates through the dataset and yields batches of images and labels.
    It also applies optional shuffling and ensures label padding for uniform batch sizes.

    Args:
        dataset (Dataset): An iterable dataset where each item is a tuple.
        batch_size (int, optional): Number of samples per batch. 
        shuffle (bool, optional): Whether to shuffle the dataset before batching. 

    Yields:
        tuple: A batch containing:
            - jnp.ndarray: Stacked batch of image tensors.
            - jnp.ndarray: Padded batch of label tensors.
            - jnp.ndarray: Array of original label lengths.
    """

    # Generate a list of indices for the dataset
    indices = list(range(len(dataset)))

    # Shuffle indices if required
    if shuffle:
        random.shuffle(indices)

    # Iterate through indices in batch-sized chunks
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]  # Select indices for the current batch

        # Retrieve batch samples from the dataset, filtering out None values
        batch_samples = [dataset[i] for i in batch_indices if dataset[i] is not None]

        if len(batch_samples) == 0:
            continue  

        # Separate images and labels from batch samples
        images, labels = zip(*batch_samples)

        # Stack images into a single tensor 
        images = jnp.stack(images)

        # Determine the maximum label length in the batch
        max_label_length = max(len(label) for label in labels)

        # Pad all labels to the same length using zero-padding
        padded_labels = jnp.array([
            jnp.pad(label, (0, max_label_length - len(label)), constant_values=0)  
            for label in labels
        ])

        yield images, padded_labels, jnp.array([len(label) for label in labels])


class CRNN(nn.Module):
    """
    A Convolutional Recurrent Neural Network (CRNN) for handwritten text recognition.

    This model consists of convolutional layers for feature extraction, bidirectional LSTM layers 
    for sequential modeling, and a fully connected layer for classification

    Attributes:
        img_height (int): The height of the input image.
        num_classes (int): The number of output classes (characters).
        lstm_hidden_size (int): Number of hidden units in the LSTM layers. Defaults to 512.
        num_lstm_layers (int): Number of LSTM layers. Defaults to 2.
    """

    img_height: int
    num_classes: int
    lstm_hidden_size: int = 512  # Default LSTM hidden size
    num_lstm_layers: int = 2  # Default number of LSTM layers

    def setup(self):
        """
        Initializes the layers of the CRNN model.
        
        - Five convolutional layers for feature extraction.
        - Batch normalization and dropout for regularization.
        - Bidirectional LSTM layers for sequence modeling.
        """
        # Convolutional layers with increasing feature depth
        self.conv1 = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding='SAME')
        self.conv2 = nn.Conv(features=128, kernel_size=(3, 3), strides=1, padding='SAME')
        self.conv3 = nn.Conv(features=256, kernel_size=(3, 3), strides=1, padding='SAME')
        self.conv4 = nn.Conv(features=256, kernel_size=(3, 3), strides=1, padding='SAME')
        self.conv5 = nn.Conv(features=512, kernel_size=(3, 3), strides=1, padding='SAME')

        # Fully connected layer for classification
        self.fc = nn.Dense(features=self.num_classes)

        # Bidirectional LSTM layers
        self.lstm_fw = [nn.LSTMCell(features=self.lstm_hidden_size) for _ in range(self.num_lstm_layers)]
        self.lstm_bw = [nn.LSTMCell(features=self.lstm_hidden_size) for _ in range(self.num_lstm_layers)]

    @nn.compact
    def __call__(self, x, train=True):
        """
        Forward pass of the CRNN model.

        Args:
            x (jnp.ndarray): Input tensor of shape (batch_size, height, width, channels).
            train (bool, optional): Whether the model is in training mode. Defaults to True.

        Returns:
            jnp.ndarray: Log softmax output of shape (batch_size, sequence_length, num_classes).
        """
        # Apply convolutional layers with ReLU activation, batch normalization, and dropout
        x = nn.relu(self.conv1(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)  # Dropout for regularization
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')  # Max pooling to downsample

        x = nn.relu(self.conv2(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        x = nn.relu(self.conv3(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        x = nn.relu(self.conv4(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)

        x = nn.relu(self.conv5(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.3, deterministic=not train)(x)

        # Get shape information
        b, h, w, c = x.shape  # Batch size, height, width, channels

        # Reshape for LSTM input 
        x = x.reshape(b, w, h * c) 

        # Initialize LSTM hidden states
        carries_fw = [lstm.initialize_carry(jax.random.PRNGKey(0), (b, self.lstm_hidden_size)) for lstm in self.lstm_fw]
        carries_bw = [lstm.initialize_carry(jax.random.PRNGKey(0), (b, self.lstm_hidden_size)) for lstm in self.lstm_bw]

        # Bidirectional LSTM processing
        outputs = []
        for t in range(x.shape[1]):  # Iterate over sequence length
            for i, lstm in enumerate(self.lstm_fw):
                carries_fw[i], x_t_fw = lstm(carries_fw[i], x[:, t, :])  # Forward LSTM processing
            for i, lstm in enumerate(self.lstm_bw):
                carries_bw[i], x_t_bw = lstm(carries_bw[i], x[:, x.shape[1] - t - 1, :])  # Backward LSTM processing
            outputs.append(jnp.concatenate([x_t_fw, x_t_bw], axis=-1))  # Concatenate bidirectional outputs

        # Convert list of tensors into a single stacked tensor 
        lstm_out = jnp.stack(outputs, axis=1)

        # Apply the fully connected layer
        output = self.fc(lstm_out)

        # Return log softmax predictions for classification
        return jax.nn.log_softmax(output, axis=-1)


def loss_fn_with_batch_stats(params, images, labels):
    """
    Computes the loss for a batch of images and labels using the CTC (Connectionist Temporal Classification) loss.

    This function applies the model with the given parameters and batch statistics, 
    updates batch normalization statistics, and calculates the loss.

    Args:
        params (dict): Model parameters.
        images (jnp.ndarray): Input images.
        labels (jnp.ndarray): Ground truth labels, where 0 represents padding.

    Returns:
        tuple:
            - loss (jnp.ndarray): The computed loss value.
            - new_batch_stats (dict): Updated batch normalization statistics.
    """

    # Forward pass with batch normalization statistics and dropout
    logits, new_model_state = model.apply(
        {'params': params, 'batch_stats': batch_stats},  
        images, 
        train=True,  
        mutable=['batch_stats'],  
        rngs={'dropout': rng}  
    )

    # Create padding masks for logits and labels
    logit_paddings = jnp.zeros((logits.shape[0], logits.shape[1]), dtype=jnp.int32)  # No padding for logits
    label_paddings = (labels == 0).astype(jnp.int32)  # Identify padded elements in labels

    # Compute CTC loss and take the mean
    loss = optax.ctc_loss(logits, logit_paddings, labels, label_paddings).mean()

    # Return loss and updated batch statistics
    return loss, new_model_state['batch_stats']


@jax.jit  # JIT compilation for efficient execution
def train_step(params, batch_stats, opt_state, images, labels):
    """
    This function performs a single training step for a neural network using Connectionist Temporal Classification (CTC) loss.

    Args:
        params (dict): Model parameters.
        batch_stats (dict): Running statistics for batch normalization.
        opt_state (optax.OptState): Optimizer state.
        images (jnp.ndarray): Input batch of images with shape.
        labels (jnp.ndarray): Ground truth labels represented as character indices.

    Returns:
        tuple:
            - params (dict): Updated model parameters.
            - new_batch_stats (dict): Updated batch normalization statistics.
            - opt_state (optax.OptState): Updated optimizer state.
            - loss (jnp.ndarray): Computed loss value for the batch.
    """

    def loss_fn_with_batch_stats(params, images, labels):
        # Perform forward pass with model, updating batch normalization statistics
        logits, new_model_state = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            images,
            train=True,
            mutable=['batch_stats'],  
            rngs={'dropout': rng} 
        )

        # Generate padding masks for logits and labels
        logit_paddings = jnp.zeros((logits.shape[0], logits.shape[1]), dtype=jnp.int32)  # No padding for logits
        label_paddings = (labels == 0).astype(jnp.int32)  # Identify padding in labels

        # Compute CTC loss and return loss along with updated batch statistics
        loss = optax.ctc_loss(logits, logit_paddings, labels, label_paddings).mean()
        return loss, new_model_state['batch_stats']

    # Compute gradients with respect to the loss function
    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn_with_batch_stats, has_aux=True)(params, images, labels)

    # Apply gradient clipping to prevent exploding gradients
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0) if g is not None else 0, grads)

    # Compute parameter updates using the optimizer
    updates, opt_state = optimizer.update(grads, opt_state)

    # Apply updates to model parameters
    params = optax.apply_updates(params, updates)

    # Return updated parameters, batch statistics, optimizer state, and loss value
    return params, new_batch_stats, opt_state, loss


def compute_cer(preds, targets):
    """
    Computes the Character Error Rate (CER) between predicted and target sequences.

    The CER is calculated as:
        CER = (Number of character errors) / (Total number of characters in the target)

    Args:
        preds (list of list of int): Predicted sequences as lists of character indices.
        targets (list of list of int): Ground truth sequences as lists of character indices.

    Returns:
        float: The computed CER value. 
    """

    total_chars, total_errors = 0, 0  # Initialize counters

    for pred, target in zip(preds, targets):
        # Convert predicted indices to text
        pred_text = ''.join([index_to_char[i] for i in pred if i in index_to_char])

        # Convert target indices to text
        target_text = ''.join([index_to_char[i] for i in target if i in index_to_char])

        # Update total character count from the ground truth
        total_chars += len(target_text)

        # Compute edit distance and update error count
        total_errors += editdistance.eval(pred_text, target_text)

    # Compute CER: ratio of total errors to total characters
    return total_errors / total_chars if total_chars > 0 else float("inf")

def greedy_decode(preds):
    """
    This function decodes a sequence of predicted character indices into a readable string 
    using the greedy decoding method for CTC-based models.

    Args:
        preds (list of int): A sequence of predicted character indices.

    Returns:
        str: The decoded text output as a string.
    """

    decoded = []  # Stores the decoded characters
    prev_char = None  # Tracks the previous character to remove duplicates

    for i in preds:
        if i == 0:  # Skip blank characters 
            prev_char = None
            continue

        if i != prev_char:  # Avoid consecutive duplicate characters 
            decoded.append(index_to_char.get(i, "?"))  # Use '?' for unknown characters

        prev_char = i  # Update previous character

    return "".join(decoded).strip()  # Join decoded characters and trim spaces


def evaluate_model(dataset):
    """
    This function evaluates the model on a given dataset by computing the average loss and Character Error Rate (CER).

    Args:
        dataset (Dataset): The dataset containing image-label pairs.

    Returns:
        tuple:
            - avg_loss (float): The average loss over the dataset.
            - cer (float): The computed Character Error Rate (CER).
    """

    total_loss, batch_count = 0, 0  # Initialize loss and batch counter
    all_preds, all_targets = [], []  # Lists to store predictions and targets

    # Iterate through dataset 
    for images, labels, label_lengths in jax_dataloader(dataset, batch_size=32, shuffle=False):
        # Compute loss for the batch
        loss, _ = loss_fn_with_batch_stats(params, images, labels)  # Correct loss function call
        total_loss += float(loss)  # Convert loss to float for accumulation
        batch_count += 1  # Update batch counter

        # Perform forward pass to get model predictions
        logits = model.apply({'params': params, 'batch_stats': batch_stats}, images, train=False, mutable=False)

        # Convert logits to character indices by selecting the most probable class
        preds = jnp.argmax(logits, axis=-1).tolist()

        # Store predictions and target labels
        all_preds.extend(preds)
        all_targets.extend(labels.tolist())

    # Print a few decoded results for qualitative evaluation
    for i in range(min(10, len(all_preds))):  # Print up to 10 decoded samples
        pred_text = greedy_decode(all_preds[i])  # Decode predicted sequence
        target_text = "".join([index_to_char.get(c, "?") for c in all_targets[i] if c > 0])  # Decode target sequence

        print(f"Decoded Target: {target_text}")
        print(f"Decoded Prediction: {pred_text}")
        print("=" * 40)

    # Compute Character Error Rate (CER)
    cer = compute_cer(all_preds, all_targets)

    # Compute average loss over all batches
    avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")

    # Return computed metrics
    return avg_loss, cer


def train_model(train_dataset, val_dataset, epochs=10):
    """
    This function trains the model for a specified number of epochs and evaluates it on the validation dataset.

    Args:
        train_dataset (Dataset): The dataset used for training.
        val_dataset (Dataset): The dataset used for validation.
        epochs (int, optional): Number of training epochs. 

    Returns:
        None
    """

    global params, batch_stats, opt_state  # Use global variables for model state

    for epoch in range(epochs):
        epoch_loss, batch_count = 0, 0  # Initialize loss and batch counter

        # Iterate through training dataset in batches
        for images, labels, label_lengths in jax_dataloader(train_dataset, batch_size=32):
            # Perform a training step: update model parameters and compute loss
            params, batch_stats, opt_state, loss = train_step(params, batch_stats, opt_state, images, labels)
            epoch_loss += float(loss)  # Accumulate batch loss
            batch_count += 1  # Update batch counter

        # Compute the average training loss for the epoch
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")

        # Evaluate the model on the validation dataset 
        val_loss, cer = evaluate_model(val_dataset)

        # Print training progress
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, CER: {cer:.4f}")
