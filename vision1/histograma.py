import numpy as np
import matplotlib.pyplot as plt
import cv2

def display_img(img):
    fig=plt.figure(figsize=(10,10))
    ax= fig.add_subplot(111)
    ax.imshow(img)

def plot_hist(img):
    color=('b','g','r')
    plt.figure()
    for i,col in enumerate(color):
        hist=cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color=col)
        plt.xlim([0,256])
    plt.title('Histograma')
    plt.show()



perro = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/perro.jpg')
ladrillos = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/ladrillos.jpg')
crucigrama = cv2.imread('C:/Users/valef/Downloads/vision1/Archivos del curso/crucigrama.jpg') 

perroRGB=cv2.cvtColor(perro,cv2.COLOR_BGR2RGB)
ladrillosRGB=cv2.cvtColor(ladrillos,cv2.COLOR_BGR2RGB)
crucigramaRGB=cv2.cvtColor(crucigrama,cv2.COLOR_BGR2RGB)

display_img(perroRGB)
display_img(ladrillosRGB)
display_img(crucigramaRGB)

plot_hist(crucigrama)

