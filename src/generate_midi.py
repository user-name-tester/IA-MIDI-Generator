import numpy as np
import tensorflow as tf
import pretty_midi
import pickle
import random
import ast
import re
from tensorflow.keras.models import load_model

# Cargar el tokenizer
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def clean_numpy_str(s):
    s = re.sub(r'np\.(float64|int32)\((.*?)\)', r'\2', s)
    return s

index_to_token = {v: ast.literal_eval(clean_numpy_str(k)) for k, v in tokenizer.items()}

print(f"Tamaño del tokenizer: {len(tokenizer)}")
print(f"Primeros 5 elementos de index_to_token: {list(index_to_token.items())[:5]}")

def generate_notes(model, seed_sequence, num_generate=200):
    generated = list(seed_sequence)
    sequence_length = len(seed_sequence)
    
    for i in range(num_generate):
        input_seq = np.array(generated[-sequence_length:]).reshape(1, sequence_length)
        preds = model.predict(input_seq, verbose=0)[0]
        next_token = np.random.choice(range(len(tokenizer)), p=preds)
        generated.append(next_token)
        # Imprimir las primeras 10 predicciones para depuración
        if i < 10:
            print(f"Predicción {i+1}: {next_token}")
    return generated

def create_midi_from_sequence(note_sequence, output_path, bpm=None):
    pm = pretty_midi.PrettyMIDI()
    
    instruments = {}
    start = 0
    
    if bpm is None:
        bpm = random.randint(80, 160)
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)  # Aplicar BPM
    
    print(f"Tokens generados (primeros 10): {note_sequence[:10]}")
    print(f"Duraciones iniciales:")
    for i, token in enumerate(note_sequence[:10]):
        if token in index_to_token:
            chord = index_to_token[token]
            max_duration = max(note_data[2] for note_data in chord)
            print(f"Token {token} (pos {i}): {chord}, duración máxima: {max_duration}")
            for note_data in chord:
                pitch, velocity, duration, instrument, channel = note_data
                if instrument not in instruments:
                    instruments[instrument] = pretty_midi.Instrument(program=instrument)
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start+duration)
                instruments[instrument].notes.append(note)
            start += max_duration
        else:
            print(f"Token {token} no encontrado en index_to_token")
    print(f"Tiempo inicial acumulado después de 10 tokens: {start} segundos")

    for token in note_sequence[10:]:
        if token in index_to_token:
            chord = index_to_token[token]
            for note_data in chord:
                pitch, velocity, duration, instrument, channel = note_data
                if instrument not in instruments:
                    instruments[instrument] = pretty_midi.Instrument(program=instrument)
                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=start+duration)
                instruments[instrument].notes.append(note)
            start += max(note_data[2] for note_data in chord)
    
    for inst in instruments.values():
        pm.instruments.append(inst)
    
    pm.write(output_path)
    print(f"MIDI generado con BPM {bpm} en: {output_path}")
    print(f"Duración total estimada: {start} segundos")

if __name__ == "__main__":
    model = load_model('model/midi_generator_model.h5')
    # Semilla variada en lugar de repetitiva
    seed = [random.randint(0, len(tokenizer) - 1) for _ in range(50)]
    print(f"Semilla inicial (primeros 10): {seed[:10]}")
    generated_notes = generate_notes(model, seed, num_generate=200)
    create_midi_from_sequence(generated_notes, 'generated/generated_song.mid')