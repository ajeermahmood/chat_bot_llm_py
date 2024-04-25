import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load your custom language model
custom_model = tf.keras.models.load_model('custom_llm_model.h5')

# Generate text samples
input_text = "What is real estate in dubai?"
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([input_text])
input_ids = tokenizer.texts_to_sequences([input_text])
input_ids_padded = pad_sequences(input_ids, maxlen=50, padding='post', truncating='post')

# Ensure input sequence has length 50 by padding
input_ids_padded = pad_sequences(input_ids, maxlen=50, padding='post', truncating='post')

# Print shapes for debugging
print("Input shape before prediction:", input_ids_padded.shape)

# Reshape input for the LSTM layer
input_ids_3d = tf.expand_dims(input_ids_padded, axis=-1)

# Define a function for predicting the next token using the LSTM layer
@tf.function
def lstm_predict_step(input_data):
    return custom_model.get_layer('lstm')(input_data)

# Apply the function to get the LSTM output
lstm_output = lstm_predict_step(input_ids_3d)

# Continue with the rest of the prediction
output_ids = custom_model.get_layer('dense')(lstm_output)

# Print shapes for debugging
print("Output shape after prediction:", output_ids.shape)
