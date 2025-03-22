import numpy as np
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint  # Importar el callback

# Cargar datos preprocesados
notes_int = np.load('notes_int.npy', allow_pickle=True)
note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
notes_set = np.load('notes_set.npy', allow_pickle=True)

# Crear secuencias de entrada para la red neuronal (ventanas de notas)
sequence_length = 100
X, y = [], []

for i in range(len(notes_int) - sequence_length):
    X.append(notes_int[i:i+sequence_length])
    y.append(notes_int[i+sequence_length])

X = np.array(X)
y = np.array(y)
y = np_utils.to_categorical(y, num_classes=len(notes_set))  # One-hot encode the target variable


# Reshape de X para que sea (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Normalizar las notas a una escala de 0 a 1
X = X / float(len(notes_set))
y = np.expand_dims(y, axis=-1)

# Cargar el modelo preentrenado si existe
try:
    # Aquí buscamos el último modelo guardado basado en los checkpoints
    model = load_model('model/model_checkpoint_epoch_latest.h5')
    print("Modelo cargado desde el último checkpoint.")
except:
    # Si no existe el modelo, crear uno nuevo
    print("No se encontró un modelo guardado. Creando uno nuevo.")
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dropout(0.3))
    model.add(Dense(len(notes_set), activation='softmax'))  # Salida con tamaño igual al número de notas únicas

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Crear el callback para guardar el modelo en cada época
checkpoint_callback = ModelCheckpoint('model/model_checkpoint_epoch_{epoch}.h5', 
                                      save_best_only=False, 
                                      save_weights_only=False, 
                                      verbose=1)

# Entrenar el modelo con el callback, comenzando desde la última época si el modelo fue cargado
model.fit(X, y, epochs=100, batch_size=64, callbacks=[checkpoint_callback])

# Al final del entrenamiento, guardar el modelo final
model.save('model/model_final.h5')
print("Entrenamiento completo y modelo guardado.")
