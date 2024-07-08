import gradio as gr 
import tensorflow as tf 
import numpy as np 

modelo = tf.keras.models.load_model("mnist_model.h5")

def clasificar_imagenes(img):
    """
    Predice el dígito en base a una imagen de 28x28 píxeles.
    
    Args -> img: np.array
    Returns -> str: dígito predicho
    """
    img = np.reshape(img, (1, 28, 28, 1)).astype("float32") / 255
    predicciones = modelo.predict(img)
    digito_predicho = np.argmax(predicciones)
    return str(digito_predicho)

# Creamos la interfaz de usuario para usar el modelo
interfaz = gr.Interface(fn=clasificar_imagenes , inputs="sketchpad", outputs="label")
interfaz.launch()