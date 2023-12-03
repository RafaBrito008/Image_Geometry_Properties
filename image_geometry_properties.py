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
        self.canvas.pack()  # Empaquetar el lienzo en la ventana principal

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
        # Método para procesar la imagen cargada
        img = cv2.imread(self.image_path)  # Leer la imagen usando OpenCV
        threshold_value = 170  # Valor de umbral para la binarización

        # Convertir la imagen a escala de grises
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Binarizar la imagen
        _, imBn = cv2.threshold(imgGray, threshold_value, 255, cv2.THRESH_BINARY)
        # Invertir la imagen binarizada
        imBn = cv2.bitwise_not(imBn)
        # Encontrar contornos en la imagen binarizada
        contornos, _ = cv2.findContours(
            imBn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filtrar contornos que son demasiado pequeños
        min_contour_area = 150  # Umbral mínimo para el área del contorno
        # Filtrar contornos por área
        contornos = [
            cnt for cnt in contornos if cv2.contourArea(cnt) > min_contour_area
        ]

        # Ordenar contornos por área en orden descendente
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        # Crear una figura de Matplotlib con 4 subplots
        fig, axs = plt.subplots(2, 2)

        # Mostrar la imagen original en el primer subplot
        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Imagen Original")

        # Mostrar la imagen binarizada con contornos en el segundo subplot
        axs[0, 1].imshow(imBn, cmap="gray")
        for contorno in contornos:
            # Dibujar cada contorno
            axs[0, 1].plot(contorno[:, 0, 0], contorno[:, 0, 1], "r", linewidth=1)
        axs[0, 1].set_title("Escala binaria con contornos")

        # Mostrar la imagen en escala de grises en el tercer subplot
        axs[1, 0].imshow(imgGray, cmap="gray")
        axs[1, 0].set_title("Escala de grises")

        # Mostrar la imagen original con contornos y centroides en el cuarto subplot
        axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for contorno in contornos:
            # Dibujar cada contorno
            axs[1, 1].plot(contorno[:, 0, 0], contorno[:, 0, 1], "r", linewidth=1)
            # Calcular y dibujar el centroide de cada contorno
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                axs[1, 1].plot(x, y, "r*", markersize=10)
        axs[1, 1].set_title("Contornos y centroides")

        # Ajustar etiquetas y mostrar la figura en el lienzo de Tkinter
        for ax in axs.flat:
            ax.label_outer()

        # Crear un canvas para Matplotlib en Tkinter y mostrar la figura
        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()  # Empaquetar el widget del canvas
        canvas.draw()  # Dibujar el canvas


# Bloque principal para ejecutar la aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()