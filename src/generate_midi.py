import numpy as np
import mido
from tensorflow.keras.models import load_model

# Cargar el modelo entrenado
model = load_model('model/model.h5')

# Cargar los datos preprocesados
note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
notes_set = np.load('notes_set.npy', allow_pickle=True)
int_to_note = {i: note for note, i in note_to_int.items()}

# Función para generar nuevas notas
def generate_notes(model, start_sequence, note_to_int, int_to_note, length=500):
    prediction = list(start_sequence)
    for _ in range(length):
        # Convertir la secuencia actual a un formato adecuado
        prediction_input = np.array(prediction)
        prediction_input = prediction_input.reshape((1, len(prediction), 1))
        prediction_input = prediction_input / float(len(note_to_int))

        # Predecir la siguiente nota
        predicted_note = model.predict(prediction_input)
        index = np.argmax(predicted_note)
        prediction.append(index)

    # Convertir las notas predichas de nuevo a notas MIDI
    predicted_notes = [int_to_note[i] for i in prediction]
    return predicted_notes

# Función para crear un archivo MIDI
def create_midi(predicted_notes, output_file='generated_song.mid'):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    for note in predicted_notes:
        track.append(mido.Message('note_on', note=note, velocity=64, time=500))  # Ajusta el tiempo según sea necesario
        track.append(mido.Message('note_off', note=note, velocity=64, time=500))

    mid.save(output_file)
    print(f"Archivo MIDI generado: {output_file}")

# Generar notas y guardar en un archivo MIDI
start_sequence = [note_to_int[note] for note in ['C4', 'E4', 'G4']]  # Semilla inicial (ajusta según quieras)
predicted_notes = generate_notes(model, start_sequence, note_to_int, int_to_note, length=500)
create_midi(predicted_notes, 'generated/generated_song.mid')
