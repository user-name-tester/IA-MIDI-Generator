# src/model_training.py
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.optimizers import Adam



def load_preprocessed_data(data_dir):
    data = []
    tempos = []
    token_to_index = {}
    
    # Primera pasada: construir el vocabulario completo
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), 'rb') as f:
                features = pickle.load(f)
                if 'notes' not in features:
                    continue
                
                for note_info in features['notes']:
                    if isinstance(note_info, list) and len(note_info) == 2:
                        _, chord = note_info
                        chord_str = str(chord)
                        if chord_str not in token_to_index:
                            token_to_index[chord_str] = len(token_to_index)
    
    # Guardar el tokenizer
    tokenizer_path = 'model/tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(token_to_index, f)
    
    # Segunda pasada: convertir los datos a índices
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), 'rb') as f:
                features = pickle.load(f)
                if 'notes' not in features:
                    continue
                
                for note_info in features['notes']:
                    if isinstance(note_info, list) and len(note_info) == 2:
                        _, chord = note_info
                        chord_str = str(chord)
                        data.append(token_to_index[chord_str])
                
                tempos.append(features['tempo'])
    
    return np.array(data), np.array(tempos), len(token_to_index)



def prepare_sequences(data, sequence_length=50):
    X = []
    y = []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Parámetros
sequence_length = 50
embedding_dim = 128
rnn_units = 256
batch_size = 64
epochs = 30

# Cargar y preparar los datos
data, tempos, vocab_size = load_preprocessed_data('preprocessed_data')

print(f"Shape of data: {data.shape}")  # Debugging line to check data shape

vocab_size = len(set(data)) if len(data) > 0 else 1

X, y = prepare_sequences(data, sequence_length)
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")  # Debugging line to check shapes of X and y


print(f"Datos de entrada: {X.shape}, Salida: {y.shape}")

data, tempos, vocab_size = load_preprocessed_data('preprocessed_data')
print(f"Vocab size: {vocab_size}")

# Definir el modelo después de conocer vocab_size
vocab_size = len(set(data)) if len(data) > 0 else 1
if vocab_size == 0:
    raise ValueError("El tamaño del vocabulario es 0. Verifica tus datos.")

print(f"Valor máximo en X: {np.max(X)}, Valor mínimo en X: {np.min(X)}")
assert np.max(X) < vocab_size, "Hay valores en X que exceden el tamaño del vocabulario. Verifica tus datos."


#definir el modelo
# Modifica la capa de Embedding para usar exactamente vocab_size
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),  # Eliminado input_length
    LSTM(rnn_units, return_sequences=True),
    Dropout(0.2),
    LSTM(rnn_units),
    Dense(vocab_size, activation='softmax')  # Asegúrate de que coincida con vocab_size
])


model.build(input_shape=(None, sequence_length))
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
model.summary()

# Entrenar el modelo
model.fit(X, y, batch_size=batch_size, epochs=epochs)

# Guardar el modelo entrenado
model.save('model/midi_generator_model.h5')
