import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import time

def guardar_resultados(ruta_archivo, salida_esperada, predicciones, precision, tiempo_total):
    """
    Guarda los resultados finales en un archivo de texto.
    """
    dias = int(tiempo_total / 86400)
    horas = int((tiempo_total % 86400) / 3600)
    minutos = int(((tiempo_total % 86400) % 3600) / 60)
    segundos = int(((tiempo_total % 86400) % 3600) % 60)
    
    with open(ruta_archivo, "w") as f:
        f.write("--- RESULTADOS FINALES ---\n\n")
        f.write(f"Tiempo de entrenamiento: {dias}d {horas}h {minutos}m {segundos}s\n\n")
        f.write("Salida esperada (y):\n")
        f.write(str(salida_esperada) + "\n\n")
        f.write("Predicciones del modelo:\n")
        f.write(str(predicciones) + "\n\n")
        f.write(f"Precisión del modelo: {precision * 100:.2f}%\n")
    print(f"\nResultados guardados en {ruta_archivo}")

def main():
    # --- 1. Preparar los datos ---
    X = np.array([[0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
    
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    
    X_transpuesto = X.T
    
    num_entradas = X_transpuesto.shape[1]
    num_salidas = 4 

    # --- 2. Definir la arquitectura de la red con Keras ---
    modelo = Sequential([
        # Cambiamos 'tanh' a 'relu'
        Dense(5, activation='relu', input_shape=(num_entradas,)),
        Dense(num_salidas, activation='softmax')
    ])

    # --- 3. Compilar el modelo ---
    modelo.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    # --- 4. Entrenar la red ---
    print("Iniciando el entrenamiento...")
    inicio_entrenamiento = time.time()
    # Entrenamos por menos épocas, ya que el aprendizaje será mucho más rápido
    historial = modelo.fit(X_transpuesto, y, epochs=50000, verbose=2) 
    fin_entrenamiento = time.time()
    tiempo_total = fin_entrenamiento - inicio_entrenamiento
    
    print("Entrenamiento finalizado.")

    # --- 5. Evaluar y predecir ---
    perdida, precision = modelo.evaluate(X_transpuesto, y, verbose=0)
    print(f"\nPrecisión del modelo: {precision * 100:.2f}%")

    predicciones = modelo.predict(X_transpuesto, verbose=0)
    predicciones_clase = np.argmax(predicciones, axis=1)

    print("\nSalida esperada (y):\n", y)
    print("\nPredicciones del modelo (índices de clase):\n", predicciones_clase)
    
    # --- 6. Guardar los resultados ---
    guardar_resultados("resultados.txt", y, predicciones_clase, precision, tiempo_total)
    
    # Guardar el modelo entrenado
    modelo.save('cerebro_ia_keras.h5')

    # --- 7. Visualizar el historial de pérdida ---
    plt.plot(historial.history['loss'])
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas')
    plt.title('Pérdida durante el entrenamiento (Keras)')
    plt.show()

if __name__ == "__main__":
    main()