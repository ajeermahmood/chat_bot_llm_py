from PyPDF2 import PdfReader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np

# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Example PDF file path
pdf_path = 'assets/book.pdf'

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Tokenize the text
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts([pdf_text])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences([pdf_text])

input_sequences = []
output_sequences = []

for sequence in sequences:
    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i])
        output_sequences.append(sequence[i])

# Pad sequences
max_seq_length = 50  # Adjust as needed
padded_sequences = pad_sequences(input_sequences, maxlen=max_seq_length)

# Convert to NumPy arrays
X = np.array(padded_sequences)
y = np.array(output_sequences)

# Define hyperparameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
lstm_units = 256
output_units = vocab_size

# Build the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_seq_length),
    LSTM(units=lstm_units),
    Dense(units=output_units, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Display model summary
model.summary()
model.save('custom_llm_model.h5')
