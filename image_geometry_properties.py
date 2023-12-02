import cv2
import numpy as np
import matplotlib.pyplot as plt

# Leer la imagen
img = cv2.imread('imagen.jpeg')

# Convertir la imagen a escala de grises
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarizar la imagen
_, imBn = cv2.threshold(imgGray, 128, 255, cv2.THRESH_BINARY)

# Invertir la imagen binaria
imBn = cv2.bitwise_not(imBn)

# Encontrar contornos
contornos, _ = cv2.findContours(imBn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Ordenar contornos por área en orden descendente
contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

# Tomar hasta un máximo de 5 objetos
numObjetosMostrados = min(5, len(contornos))

# Mostrar la imagen original
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Imagen Original')

# Mostrar la imagen binaria con contornos
plt.subplot(2, 2, 2)
plt.imshow(imBn, cmap='gray')
for i in range(numObjetosMostrados):
    plt.plot(contornos[i][:, 0, 0], contornos[i][:, 0, 1], 'r', linewidth=1)
plt.title('Escala binaria con contornos')

# Mostrar la escala de grises
plt.subplot(2, 2, 3)
plt.imshow(imgGray, cmap='gray')
plt.title('Escala de grises')

# Mostrar contornos y centroides
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
for i in range(numObjetosMostrados):
    plt.plot(contornos[i][:, 0, 0], contornos[i][:, 0, 1], 'r', linewidth=1)
    
    M = cv2.moments(contornos[i])
    if M['m00'] != 0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])
        plt.plot(x, y, 'r*', markersize=10)

plt.title('Contornos y centroides para los primeros 5 objetos de mayor área')
plt.show()
