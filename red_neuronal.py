import numpy as np

class RedNeuronal:
    """
    Clase para construir y entrenar una red neuronal desde cero.
    """
    def __init__(self, tamano_capas, activaciones):
        """
        Inicializa la red neuronal con una arquitectura dada.
        
        Args:
            tamano_capas (list): Lista de enteros que define el número de neuronas
                                 en cada capa. Ej: [2, 3, 1] para 2 entradas,
                                 una capa oculta de 3 neuronas y 1 salida.
            activaciones (list): Lista de nombres de funciones de activación 
                                 para cada capa oculta y de salida.
        """
        self.tamano_capas = tamano_capas
        self.activaciones = activaciones
        self.pesos = []
        self.sesgos = []
        
        for i in range(1, len(self.tamano_capas)):
            w = np.random.randn(self.tamano_capas[i], self.tamano_capas[i-1]) * 0.01
            self.pesos.append(w)
            
            b = np.zeros((self.tamano_capas[i], 1))
            self.sesgos.append(b)

    def _funcion_activacion(self, Z, nombre_funcion):
        """
        Aplica una función de activación a una matriz Z.
        """
        if nombre_funcion == "relu":
            return np.maximum(0, Z)
        elif nombre_funcion == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        elif nombre_funcion == "tanh":
            return np.tanh(Z)
        elif nombre_funcion == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            raise ValueError(f"Función de activación '{nombre_funcion}' no soportada.")

    def _derivada_activacion(self, A, nombre_funcion):
        """
        Calcula la derivada de una función de activación.
        """
        if nombre_funcion == "relu":
            return np.where(A > 0, 1, 0)
        elif nombre_funcion == "sigmoid":
            return A * (1 - A)
        elif nombre_funcion == "tanh":
            return 1 - np.square(A)
        else:
            raise ValueError(f"Función de activación '{nombre_funcion}' no soportada.")
    
    def _entropia_cruzada_binaria(self, A, Y):
        """
        Calcula la pérdida usando la entropía cruzada binaria.
        """
        m = Y.shape[1]
        costo = -np.sum(Y * np.log(A + 1e-10) + (1 - Y) * np.log(1 - A + 1e-10)) / m
        return np.squeeze(costo)

    def _entropia_cruzada_categorica(self, A, Y):
        """
        Calcula la pérdida usando la entropía cruzada categórica.
        """
        m = Y.shape[1]
        costo = -np.sum(Y * np.log(A + 1e-10)) / m
        return np.squeeze(costo)

    def _propagacion_atras(self, caches, Y):
        """
        Calcula los gradientes para la propagación hacia atrás.
        """
        m = Y.shape[1]
        gradientes = {}
        
        A_final = caches[f"A{len(self.tamano_capas) - 1}"]
        dZ_actual = A_final - Y

        for i in reversed(range(len(self.tamano_capas) - 1)):
            A_previa = caches[f"A{i}"]
            
            dW = np.dot(dZ_actual, A_previa.T) / m
            db = np.sum(dZ_actual, axis=1, keepdims=True) / m
            
            gradientes[f"dW{i+1}"] = dW
            gradientes[f"db{i+1}"] = db
            
            if i > 0:
                dA_previa = np.dot(self.pesos[i].T, dZ_actual)
                nombre_activacion = self.activaciones[i - 1]
                A_anterior = caches[f"A{i}"]
                dZ_actual = dA_previa * self._derivada_activacion(A_anterior, nombre_activacion)
            
        return gradientes

    def propagacion_adelante(self, X):
        """
        Calcula la salida de la red neuronal para una entrada X.
        """
        caches = {"A0": X}
        A = X
        
        for i in range(len(self.tamano_capas) - 1):
            pesos_actuales = self.pesos[i]
            sesgos_actuales = self.sesgos[i]
            nombre_activacion = self.activaciones[i]
            
            Z = np.dot(pesos_actuales, A) + sesgos_actuales
            A = self._funcion_activacion(Z, nombre_activacion)
            
            caches[f"Z{i+1}"] = Z
            caches[f"A{i+1}"] = A
        
        return caches

    def entrenar(self, X, y, epocas, tasa_aprendizaje):
        """
        Entrena la red neuronal con datos de entrada y etiquetas.
        """
        costos = []
        
        print("Iniciando el entrenamiento...")
        for i in range(epocas):
            caches = self.propagacion_adelante(X)
            salida_final = caches[f"A{len(self.tamano_capas) - 1}"]
            
            costo = self._entropia_cruzada_categorica(salida_final, y)
            
            if i % 10000 == 0:
                costos.append(costo)
            
            gradientes = self._propagacion_atras(caches, y)
            
            for j in range(len(self.pesos)):
                self.pesos[j] -= tasa_aprendizaje * gradientes[f"dW{j+1}"]
                self.sesgos[j] -= tasa_aprendizaje * gradientes[f"db{j+1}"]
            
            if i % 10000 == 0:
                print(f"Época {i}: Pérdida = {costo:.4f}")

        print("Entrenamiento finalizado.")
        return costos
    
    def guardar_modelo(self, ruta_archivo):
        """
        Guarda los pesos y sesgos del modelo en un archivo .npz.
        """
        print(f"Guardando el modelo en {ruta_archivo}...")
        
        datos_guardar = {
            'tamano_capas': self.tamano_capas,
            'activaciones': self.activaciones
        }
        
        for i, peso in enumerate(self.pesos):
            datos_guardar[f'pesos_{i}'] = peso
            
        for i, sesgo in enumerate(self.sesgos):
            datos_guardar[f'sesgos_{i}'] = sesgo
            
        np.savez(ruta_archivo, **datos_guardar)
        
        print("Modelo guardado exitosamente.")
    
    def cargar_modelo(self, ruta_archivo):
        """
        Carga los pesos y sesgos de un archivo .npz.
        """
        print(f"Cargando el modelo desde {ruta_archivo}...")
        
        datos = np.load(ruta_archivo, allow_pickle=True)
        
        self.tamano_capas = datos['tamano_capas'].tolist()
        self.activaciones = datos['activaciones'].tolist()
        
        self.pesos = []
        self.sesgos = []
        
        for i in range(len(self.tamano_capas) - 1):
            self.pesos.append(datos[f'pesos_{i}'])
            self.sesgos.append(datos[f'sesgos_{i}'])
        
        print("Modelo cargado exitosamente.")

    def predecir(self, X):
        """
        Realiza una predicción para una o más entradas X.
        """
        caches = self.propagacion_adelante(X)
        salida = caches[f"A{len(self.tamano_capas) - 1}"]
        
        predicciones_clase = np.argmax(salida, axis=0)

        num_clases = salida.shape[0]
        one_hot_predicciones = np.eye(num_clases)[predicciones_clase].T
        
        return one_hot_predicciones

    def evaluar_precision(self, X, y):
        """
        Calcula la precisión del modelo para las entradas X y las etiquetas y.
        """
        predicciones = self.predecir(X)
        
        aciertos = np.all(predicciones == y, axis=0).sum()
        total_ejemplos = y.shape[1]
        
        precision = aciertos / total_ejemplos
        return precision