import os
import mido
import numpy as np

# Función para extraer notas y duraciones de un archivo MIDI
def extract_notes_and_durations_from_midi(file_path):
    mid = mido.MidiFile(file_path)
    notes = []
    note_durations = []
    active_notes = {}
    
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    duration = current_time - active_notes[msg.note]
                    notes.append(msg.note)
                    note_durations.append(duration)
                    del active_notes[msg.note]
    
    return notes, note_durations

# Función para extraer el tempo de un archivo MIDI
def extract_tempo_from_midi(file_path):
    mid = mido.MidiFile(file_path)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)  # Convierte a BPM
    return 120  # Valor por defecto si no se encuentra tempo

# Cargar todos los archivos MIDI de una carpeta
def load_midi_files(folder_path):
    all_notes = []
    all_durations = []
    tempos = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.mid'):
            file_path = os.path.join(folder_path, filename)
            notes, durations = extract_notes_and_durations_from_midi(file_path)
            tempo = extract_tempo_from_midi(file_path)
            
            all_notes.extend(notes)
            all_durations.extend(durations)
            tempos.append(tempo)
    
    return all_notes, all_durations, tempos

# Preprocesamiento: convertir notas y duraciones a enteros
def preprocess_data(all_notes, all_durations):
    notes_set = sorted(set(all_notes))  # Notas únicas
    note_to_int = {note: number for number, note in enumerate(notes_set)}
    notes_int = [note_to_int[note] for note in all_notes]
    
    durations_set = sorted(set(all_durations))  # Duraciones únicas
    duration_to_int = {duration: number for number, duration in enumerate(durations_set)}
    durations_int = [duration_to_int[duration] for duration in all_durations]
    
    return notes_int, durations_int, note_to_int, duration_to_int, notes_set, durations_set

if __name__ == "__main__":
    midi_folder = os.path.join(os.path.dirname(__file__), '..', 'data')

    print("Cargando archivos MIDI desde:", midi_folder)
    all_notes, all_durations, tempos = load_midi_files(midi_folder)
    print(f"Se cargaron {len(all_notes)} notas desde los archivos MIDI.")
    
    print("Preprocesando datos...")
    notes_int, durations_int, note_to_int, duration_to_int, notes_set, durations_set = preprocess_data(all_notes, all_durations)
    print("Preprocesamiento completado.")
    
    # Guardar los datos preprocesados
    print("Guardando archivos de datos preprocesados...")
    np.save('notes_int.npy', notes_int)
    np.save('durations_int.npy', durations_int)
    np.save('note_to_int.npy', note_to_int)
    np.save('duration_to_int.npy', duration_to_int)
    np.save('notes_set.npy', notes_set)
    np.save('durations_set.npy', durations_set)
    np.save('tempos.npy', tempos)
    
    print("Todos los archivos fueron guardados exitosamente.")
