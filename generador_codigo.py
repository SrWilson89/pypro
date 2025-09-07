import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def guardar_resultados(ruta_archivo, texto_generado):
    """
    Guarda el texto generado en un archivo de texto.
    """
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        f.write("--- CÓDIGO GENERADO POR LA IA ---\n\n")
        f.write(texto_generado)
    print(f"\nCódigo generado y guardado en {ruta_archivo}")

def redireccionar_errores(ruta_archivo):
    """
    Redirige la salida de la consola (stdout y stderr) a un archivo.
    """
    sys.stdout = open(ruta_archivo, 'a', encoding="utf-8")
    sys.stderr = sys.stdout

def main():
    # Redirigimos la salida de la consola a errores.txt
    redireccionar_errores("errores.txt")
    print("Iniciando la generación de código...")
    print("Esto puede tomar unos minutos la primera vez que se descarga el modelo.")
    
    # 1. Definir el "cerebro" de la IA y el tokenizador
    # El modelo 'codeparrot/codeparrot-small-multi' es mucho más potente para generar código.
    try:
        tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small-multi")
        # El parámetro 'device_map="auto"' requiere la librería 'accelerate'
        modelo = AutoModelForCausalLM.from_pretrained("codeparrot/codeparrot-small-multi", device_map="auto")
    except Exception as e:
        print(f"Error al descargar el modelo o el tokenizador: {e}")
        print("\nPor favor, asegúrate de tener PyTorch o TensorFlow instalado.")
        print("Puedes hacerlo con 'pip install torch' o 'pip install tensorflow'.")
        return

    # 2. La tarea que le daremos a la IA
    prompt = """
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

# La IA debe completar esta función:
def es_par(numero):
    """
    
    # 3. Convertir el texto en algo que la IA pueda entender
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # 4. Usar el "cerebro" para generar código
    output = modelo.generate(
        input_ids,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # 5. Decodificar el código generado por la IA
    texto_generado = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 6. Guardar el resultado
    guardar_resultados("resultados.txt", texto_generado)
    
    print("\nProceso completado. Revisa 'resultados.txt' y 'errores.txt'.")

if __name__ == "__main__":
    main()