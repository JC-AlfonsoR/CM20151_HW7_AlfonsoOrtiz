#album_lento.py
from numpy import *
import datetime

#Defino la funcion
def llenar_album(n):
    # n Numero de laminas
    # Devuelve el numero de laminas que hay que comprar para llenar el album
    album = zeros(n)
    n_album = 0
    l_compradas = 0
    while n_album < n:
        mona = random.randint(low=0, high=n) # Compro lamina
        l_compradas += 1
        if album[mona] == 0: # Si no la tengo, la agrego al album
            album[mona] = 1
            n_album += 1
    return l_compradas

n = 640
#c = 1200.0/5.0
# Forma lenta de generar los datos
#print(datetime.datetime.now())
M = arange(50000,70000,10000)
Media = zeros(len(M))
Varianza = zeros(len(M))

for i in range(len(M)):
    m  = M[i] # numero de albumes a llenar
    
    N_fichas = zeros(m) # numero de fichas para llenar album
    for j in range(m):    
        N_fichas[j] = llenar_album(n) # Calcular numero de fichas
    Costo = N_fichas # Calcular costo -> Numero de fichas
    Media[i] = mean(Costo)
    Varianza[i] = var(Costo)
    
    #if m%1000 == 0:
    #    print(m,end=' ')

#Guardar datos    
Data = vstack((M,Media,Varianza)).T
savetxt('Album_lento_3.csv',Data,delimiter=',')
print(datetime.datetime.now())
