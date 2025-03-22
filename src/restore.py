from tensorflow.keras.models import save_model
import tensorflow as tf

# Verificar si el modelo está en memoria
try:
    # Intentar obtener el modelo de la sesión de TensorFlow (si está en memoria)
    model = tf.keras.backend.get_session().graph.get_tensor_by_name('model:0')
    
    # Si lo encuentra, lo guarda en el archivo
    save_model(model, 'model/model_checkpoint_in_memory.h5')
    print("Modelo guardado con éxito.")
except Exception as e:
    print("No se pudo encontrar el modelo en memoria. Error:", str(e))
