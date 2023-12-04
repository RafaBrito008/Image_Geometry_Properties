import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageProcessorApp:
    def __init__(self, root):
        # Inicializar la aplicación con la ventana principal
        self.root = root
        self.root.title("Procesador de Imágenes")  # Título de la ventana
        self.image_path = None  # Ruta de la imagen a procesar
        self.setup_ui()  # Configurar la interfaz de usuario

    def setup_ui(self):
        # Configuración de la interfaz de usuario
        self.frame = tk.Frame(self.root)  # Crear un marco en la ventana principal
        self.frame.pack(padx=10, pady=10)  # Empaquetar el marco con un poco de espacio

        # Botón para cargar imágenes
        self.button_load = tk.Button(
            self.frame, text="Cargar Imagen", command=self.load_image
        )
        self.button_load.pack(side=tk.TOP, pady=5)  # Posicionar el botón

        # Crear un lienzo para mostrar imágenes
        self.canvas = tk.Canvas(self.root, width=600, height=600)
        self.canvas.pack(
            fill=tk.BOTH, expand=True
        )  # Empaquetar el lienzo en la ventana principal

        # Vincular el evento de cierre de la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        # Método para manejar el cierre de la ventana
        self.root.quit()  # Terminar el bucle principal
        self.root.destroy()  # Destruir la ventana

    def load_image(self):
        # Método para cargar una imagen
        # Mostrar un cuadro de diálogo para elegir un archivo
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")]
        )
        if self.image_path:
            self.process_image()  # Si se selecciona una imagen, procesarla

    def process_image(self):
        # Parámetros configurables para el procesamiento de la imagen
        threshold_value = 128  # Valor para la binarización de la imagen
        min_contour_area = 150  # Área mínima para considerar un contorno

        # Cargar la imagen desde la ruta especificada. OpenCV por defecto carga las imágenes en formato BGR.
        img = cv2.imread(self.image_path)

        # Convertir la imagen de BGR (azul, verde, rojo) a escala de grises para simplificar el análisis y procesamiento.
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Aplicar un umbral binario a la imagen en escala de grises. Todos los píxeles con un valor superior al umbral se establecen en blanco (255),
        # y todos los píxeles con un valor inferior al umbral se establecen en negro (0).
        _, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

        # Invertir la imagen binaria, cambia los píxeles blancos a negros y viceversa, esto es necesario para la detección de contornos en OpenCV.
        img_binary_inv = cv2.bitwise_not(img_binary)

        # Detectar los contornos en la imagen binaria invertida. OpenCV busca contornos desde los píxeles blancos.
        contornos, _ = cv2.findcontornos(
            img_binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtrar los contornos para eliminar aquellos que son demasiado pequeños, basándose en el área mínima definida.
        contornos = [
            cnt for cnt in contornos if cv2.contourArea(cnt) > min_contour_area
        ]

        # Ordenar los contornos detectados por su área, de mayor a menor.
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        # Crear una figura con 4 subplots (2x2) para mostrar diferentes transformaciones de la imagen.
        fig, axs = plt.subplots(2, 2)

        # Mostrar la imagen original en el primer subplot.
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Imagen Original")
        axs[0, 0].axis("off")  # Ocultar los ejes para una mejor visualización.

        # Mostrar la imagen binaria invertida en el segundo subplot y dibujar los contornos encontrados.
        axs[0, 1].imshow(img_binary_inv, cmap="gray")
        axs[0, 1].set_title("Escala binaria con contornos")
        axs[0, 1].axis("off")
        for contour in contornos:
            # Dibujar el contorno en rojo sobre la imagen binaria invertida.
            axs[0, 1].plot(contour[:, 0, 0], contour[:, 0, 1], "r", linewidth=1)

        # Mostrar la imagen en escala de grises en el tercer subplot.
        axs[1, 0].imshow(img_gray, cmap="gray")
        axs[1, 0].set_title("Escala de grises")
        axs[1, 0].axis("off")

        # Mostrar la imagen original en el cuarto subplot y dibujar los contornos y centroides.
        axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[1, 1].set_title("Contornos y centroides")
        axs[1, 1].axis("off")
        for contour in contornos:
            # Dibujar el contorno en rojo sobre la imagen original.
            axs[1, 1].plot(contour[:, 0, 0], contour[:, 0, 1], "r", linewidth=1)
            # Calcular el centroide del contorno y dibujarlo como una estrella roja.
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                axs[1, 1].plot(cX, cY, "r*", markersize=10)

        # Ajustar las etiquetas de los ejes y mostrar la figura en el lienzo de Tkinter
        for ax in axs.flat:
            ax.label_outer()

        # Integrar la figura de Matplotlib en el lienzo de Tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()


# Bloque principal para ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
