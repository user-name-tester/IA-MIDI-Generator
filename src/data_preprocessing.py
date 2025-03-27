# src/data_preprocessing.py
import os
import numpy as np
import pretty_midi
import pickle

def extract_midi_features(midi_path):
    """Extrae características relevantes de un archivo MIDI, incluyendo acordes."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"Error al leer {midi_path}: {e}")
        return None

    # Extraer tempo estimado
    tempo = pm.estimate_tempo()
    
    # Extraer notas, duración, canal e instrumento, detectando acordes
    notes_info = []
    active_notes = {}
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            duration = note.end - note.start
            start_time = round(note.start, 3)  # Redondear para detectar notas simultáneas
            
            # Agrupar notas que comienzan al mismo tiempo como un acorde
            if start_time not in active_notes:
                active_notes[start_time] = []
            active_notes[start_time].append((note.pitch, note.velocity, duration, instrument.program, instrument.program % 16))
    
    # Convertir los acordes detectados en listas de notas
    for start_time, chord in sorted(active_notes.items()):
        notes_info.append([start_time, chord])  # Guardamos el acorde como lista de notas
    
    features = {
        'tempo': tempo,
        'notes': notes_info
    }
    return features

def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    all_notes = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.mid') or filename.lower().endswith('.midi'):
            midi_file = os.path.join(input_dir, filename)
            features = extract_midi_features(midi_file)
            if features is not None:
                all_notes.extend(features['notes'])
                output_file = os.path.join(output_dir, filename + '.pkl')
                with open(output_file, 'wb') as f:
                    pickle.dump(features, f)
                print(f"Procesado: {filename}")
    
    # Crear tokenizer para acordes y notas individuales
    cleaned_notes = []
    for note in all_notes:
        if isinstance(note, list):
            cleaned_notes.append(tuple(tuple(n) if isinstance(n, (list, tuple)) else (n,) for n in note))
        elif isinstance(note, (int, float, np.float64, np.int64)):
            cleaned_notes.append((float(note),))
        else:
            print(f"⚠️ Advertencia: Tipo de dato inesperado en all_notes -> {type(note)}, valor: {note}")

    unique_tokens = list(set(cleaned_notes))
    tokenizer = {i: unique_tokens[i] for i in range(len(unique_tokens))}
    with open('model/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

if __name__ == "__main__":
    process_dataset('data', 'preprocessed_data')
