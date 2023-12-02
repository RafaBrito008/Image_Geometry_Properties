import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procesador de Imágenes")
        self.image_path = None
        self.setup_ui()

    def setup_ui(self):
            self.frame = tk.Frame(self.root)
            self.frame.pack(padx=10, pady=10)

            self.button_load = tk.Button(self.frame, text="Cargar Imagen", command=self.load_image)
            self.button_load.pack(side=tk.TOP, pady=5)

            self.canvas = tk.Canvas(self.root, width=600, height=600)
            self.canvas.pack()

            self.root.protocol("WM_DELETE_WINDOW", self.on_close)  # Vincular el evento de cierre

    def on_close(self):
        self.root.quit()
        self.root.destroy()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.jpg;*.png;*.jpeg")])
        if self.image_path:
            self.process_image()

    def process_image(self):
        img = cv2.imread(self.image_path)

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, imBn = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)
        imBn = cv2.bitwise_not(imBn)
        contornos, _ = cv2.findContours(imBn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos que son demasiado pequeños
        min_contour_area = 150  # Define un umbral mínimo para el área del contorno
        contornos = [cnt for cnt in contornos if cv2.contourArea(cnt) > min_contour_area]

        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

        fig, axs = plt.subplots(2, 2)

        axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Imagen Original')

        axs[0, 1].imshow(imBn, cmap='gray')
        for contorno in contornos:
            axs[0, 1].plot(contorno[:, 0, 0], contorno[:, 0, 1], 'r', linewidth=1)
        axs[0, 1].set_title('Escala binaria con contornos')

        axs[1, 0].imshow(imgGray, cmap='gray')
        axs[1, 0].set_title('Escala de grises')

        axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        for contorno in contornos:
            axs[1, 1].plot(contorno[:, 0, 0], contorno[:, 0, 1], 'r', linewidth=1)
            M = cv2.moments(contorno)
            if M['m00'] != 0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                axs[1, 1].plot(x, y, 'r*', markersize=10)
        axs[1, 1].set_title('Contornos y centroides')

        for ax in axs.flat:
            ax.label_outer()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()
        canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()