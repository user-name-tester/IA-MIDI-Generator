import os
import mido
import numpy as np

# Función para extraer las notas de un archivo MIDI
def extract_notes_from_midi(file_path):
    mid = mido.MidiFile(file_path)
    notes = []
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on':  # Detectar cuando una nota empieza
                notes.append(msg.note)  # Nota MIDI
    return notes

# Cargar todos los archivos MIDI de una carpeta
def load_midi_files(folder_path):
    all_notes = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.mid'):
            file_path = os.path.join(folder_path, filename)
            notes = extract_notes_from_midi(file_path)
            all_notes.extend(notes)
    return all_notes

# Preprocesamiento: convertir las notas a enteros
def preprocess_data(all_notes):
    notes_set = sorted(set(all_notes))  # Conjunto de notas únicas
    note_to_int = {note: number for number, note in enumerate(notes_set)}
    notes_int = [note_to_int[note] for note in all_notes]
    return notes_int, note_to_int, notes_set

if __name__ == "__main__":
    # Cambia esto por la ruta donde tienes tus archivos MIDI
    midi_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

    all_notes = load_midi_files(midi_folder)
    notes_int, note_to_int, notes_set = preprocess_data(all_notes)
    # Guardamos los datos procesados
    np.save('notes_int.npy', notes_int)
    np.save('note_to_int.npy', note_to_int)
    np.save('notes_set.npy', notes_set)
    print("Preprocesamiento completo.")
