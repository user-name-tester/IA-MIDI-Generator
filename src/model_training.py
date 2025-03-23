import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Cargar los datos preprocesados (notas, duraciones, tempo)
notes_int = np.load('notes_int.npy', allow_pickle=True)
durations_int = np.load('durations_int.npy', allow_pickle=True)
tempo = np.load('tempos.npy', allow_pickle=True)

note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
duration_to_int = np.load('duration_to_int.npy', allow_pickle=True).item()
notes_set = np.load('notes_set.npy', allow_pickle=True)
durations_set = np.load('durations_set.npy', allow_pickle=True)

# Imprimir la longitud de notas
print(f"Longitud de notes_int: {len(notes_int)}")
print(f"Longitud de durations_int: {len(durations_int)}")
print(f"Longitud de tempo: {len(tempo)}")

# Generar secuencias de entrada (X) y salida (y)
sequence_length = 100  # Tamaño de la secuencia de entrada

X_notes = []
y_notes = []
X_durations = []
y_durations = []
X_tempos = []  # Solo guardaremos un tempo por secuencia
y_tempos = []  # Salida también debe ser 1 valor de tempo

for i in range(len(notes_int) - sequence_length):
    if i + sequence_length < len(durations_int):
        seq_in_notes = notes_int[i:i+sequence_length]
        seq_out_notes = notes_int[i+sequence_length]
        X_notes.append(seq_in_notes)
        y_notes.append(seq_out_notes)

        seq_in_durations = durations_int[i:i+sequence_length]
        seq_out_durations = durations_int[i+sequence_length]
        X_durations.append(seq_in_durations)
        y_durations.append(seq_out_durations)

        # Asegurar que tempo se usa correctamente
        tempo_value = tempo[i // (len(notes_int) // len(tempo))]  # Ajustar el índice de tempo
        X_tempos.append(tempo_value)
        y_tempos.append(tempo_value)

# Convertir a arrays de numpy
X_notes = np.reshape(X_notes, (len(X_notes), sequence_length, 1)) / float(len(notes_set))
X_durations = np.reshape(X_durations, (len(X_durations), sequence_length, 1)) / float(len(durations_set))
X_tempos = np.array(X_tempos).reshape(-1, 1) / float(max(tempo))

y_notes = tf.keras.utils.to_categorical(y_notes, num_classes=len(notes_set))
y_durations = tf.keras.utils.to_categorical(y_durations, num_classes=len(durations_set))
y_tempos = np.array(y_tempos).reshape(-1, 1)  # Mantener la estructura correcta


# Definir el modelo
input_notes = Input(shape=(X_notes.shape[1], X_notes.shape[2]))
input_durations = Input(shape=(X_durations.shape[1], X_durations.shape[2]))
input_tempos = Input(shape=(1,))  # Un solo valor por muestra

x_notes = LSTM(512, return_sequences=True)(input_notes)
x_notes = Dropout(0.3)(x_notes)
x_notes = LSTM(512)(x_notes)
x_notes = Dropout(0.3)(x_notes)

x_durations = LSTM(512, return_sequences=True)(input_durations)
x_durations = Dropout(0.3)(x_durations)
x_durations = LSTM(512)(x_durations)
x_durations = Dropout(0.3)(x_durations)

x_tempos = Dense(128, activation="relu")(input_tempos)
x_tempos = Dropout(0.3)(x_tempos)

# Fusionar las salidas
merged = tf.keras.layers.concatenate([x_notes, x_durations, x_tempos])

output_notes = Dense(len(notes_set), activation='softmax')(merged)
output_durations = Dense(len(durations_set), activation='softmax')(merged)
output_tempos = Dense(1, activation='linear')(merged)  # Predicción numérica continua

model = Model(inputs=[input_notes, input_durations, input_tempos],
              outputs=[output_notes, output_durations, output_tempos])

# Compilar el modelo
model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'mse'],  # MSE para tempo
              optimizer='adam')

# Definir el callback para guardar el modelo
checkpoint = ModelCheckpoint('model_checkpoint_epoch_latest.h5', save_best_only=True)

# Entrenar el modelo
model.fit([X_notes, X_durations, X_tempos], [y_notes, y_durations, y_tempos],
          epochs=50, batch_size=64, callbacks=[checkpoint])

# Crear carpeta "model" si no existe
os.makedirs('model', exist_ok=True)

# Guardar el modelo después de entrenar
model.save('model/model_checkpoint_epoch_latest.h5')
