import os
import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack
import pretty_midi
from tensorflow.keras.models import load_model

# Cargar los datos preprocesados
notes_int = np.load('notes_int.npy', allow_pickle=True)
durations_int = np.load('durations_int.npy', allow_pickle=True)
note_to_int = np.load('note_to_int.npy', allow_pickle=True).item()
duration_to_int = np.load('duration_to_int.npy', allow_pickle=True).item()
notes_set = np.load('notes_set.npy', allow_pickle=True)
durations_set = np.load('durations_set.npy', allow_pickle=True)
tempos = np.load('tempos.npy', allow_pickle=True)

# Cargar el modelo entrenado
model = load_model('model/model_final.h5')

# Invertir los diccionarios de mapeo para reconstrucción
int_to_note = {number: note for note, number in note_to_int.items()}
int_to_duration = {number: duration for duration, number in duration_to_int.items()}

def generate_midi(start_sequence, length=100, tempo=None):
    generated_notes = start_sequence[:]
    generated_durations = [0.5] * len(start_sequence)  # Duraciones por defecto
    
    for _ in range(length):
        input_sequence = np.array(generated_notes[-100:]).reshape(1, 100, 1) / float(len(notes_set))
        prediction = model.predict([input_sequence, input_sequence, input_sequence], verbose=0)  # Usar entradas correspondientes
        predicted_note = np.argmax(prediction[0])  # Primera salida es notas
        predicted_duration = np.argmax(prediction[1])  # Segunda salida es duraciones
        predicted_tempo = np.argmax(prediction[2])  # Tercera salida es tempo
        
        generated_notes.append(predicted_note)
        generated_durations.append(predicted_duration)

        # Puedes agregar un ajuste para cambiar el tempo si es necesario
        if tempo is None:
            tempo = predicted_tempo  # Si no se pasa tempo, usar el predicho
            
    # Crear un archivo MIDI con las notas generadas
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Definir el tempo si no se especifica
    if tempo is None:
        tempo = np.median(tempos)  # Usar la mediana de los tempos si no se pasa
    microseconds_per_beat = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
    
    # Escribir las notas y duraciones en el archivo MIDI
    for note_int, duration in zip(generated_notes, generated_durations):
        note = int_to_note.get(note_int, 60)  # Nota por defecto: C4
        duration_ticks = int(midi.ticks_per_beat * duration)
        
        # Añadir las notas al track
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=duration_ticks))
    
    # Guardar el archivo MIDI generado
    output_path = 'generated_music.mid'
    midi.save(output_path)
    print(f"Archivo MIDI generado y guardado en {output_path}")

# Ejemplo de uso
generate_midi(start_sequence=notes_int[:100], length=200, tempo=120)
