import numpy as np
from red_neuronal import RedNeuronal
import matplotlib.pyplot as plt
import time
import sys # Añadimos esta librería para saber qué sistema operativo estamos usando

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
        f.write(f"Precisión del modelo: {precision:.2f}\n")
    print(f"\nResultados guardados en {ruta_archivo}")

# Nueva función para reproducir un sonido
def reproducir_sonido_finalizacion():
    """
    Reproduce un sonido suave para avisar que el proceso ha terminado.
    """
    print("Reproduciendo sonido de finalización...")
    try:
        if sys.platform == 'win32':
            import winsound
            winsound.Beep(440, 500)  # Frecuencia 440 Hz, duración 500 ms
        elif sys.platform == 'darwin': # macOS
            import os
            os.system('say "El entrenamiento ha finalizado"')
        else: # Linux y otros
            import os
            os.system('printf "\a"') # Sonido de campana
    except Exception as e:
        print(f"No se pudo reproducir el sonido: {e}")

def main():
    # --- 1. Definir la arquitectura de la red y los datos ---
    tamano_capas = [2, 5, 4] 
    activaciones = ["tanh", "softmax"] 
    
    X = np.array([[0, 0, 1, 1, 0, 0, 1, 1],
                  [0, 1, 0, 1, 0, 1, 0, 1]])
    
    y = np.array([[1, 1, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 1, 1, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 1, 1, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 1, 1]]).astype(np.float64) 
    
    # --- 2. Fase de Entrenamiento y Visualización ---
    print("--- FASE 1: ENTRENAMIENTO Y VISUALIZACIÓN ---")
    red_entrenamiento = RedNeuronal(tamano_capas, activaciones)
    
    epocas = 1200000 
    tasa_aprendizaje = 0.1
    
    inicio_entrenamiento = time.time()
    historial_costos = red_entrenamiento.entrenar(X, y, epocas, tasa_aprendizaje)
    fin_entrenamiento = time.time()
    tiempo_total = fin_entrenamiento - inicio_entrenamiento

    # --- 3. Guardar el modelo entrenado ---
    ruta_modelo = "cerebro_ia.npz"
    red_entrenamiento.guardar_modelo(ruta_modelo)

    # --- 4. Visualizar el progreso del entrenamiento ---
    print("\nVisualizando el progreso del entrenamiento...")
    plt.plot(historial_costos)
    plt.ylabel('Pérdida')
    plt.xlabel('Épocas (x10,000)')
    plt.title('Pérdida durante el entrenamiento')
    plt.show()

    # --- 5. Fase de Carga y Predicción ---
    print("\n--- FASE 2: CARGA Y PREDICCIÓN ---")
    
    red_prueba = RedNeuronal(tamano_capas, activaciones)
    red_prueba.cargar_modelo(ruta_modelo)
    
    predicciones = red_prueba.predecir(X)
    
    print("\nSalida esperada (y):")
    print(y)
    print("\nPredicciones del modelo (predecir(X)):")
    print(predicciones)

    # --- 6. Evaluar la precisión y guardar los resultados ---
    precision = red_prueba.evaluar_precision(X, y)
    print(f"\nPrecisión del modelo: {precision:.2f}")

    guardar_resultados("resultados.txt", y, predicciones, precision, tiempo_total)

    # Llamamos a la función de sonido para que avise al terminar
    reproducir_sonido_finalizacion()
    
if __name__ == "__main__":
    main()