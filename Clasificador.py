import scipy.io as mat
import numpy as np
from scipy.signal import butter
from scipy.signal import lfilter
from scipy.signal import resample_poly
from scipy.stats import norm

#Parámetros fijos
Fs        = 250         # Frecuencia de muestreo
f_inf     = 11          # Banda inferor del filtro
f_sup     = 13          # Banda superior del filtro
t_exp     = 5.0         # Tiempo de experimento a utilizar
num_datos = 78          # Número de muestras a utilizar en el entrenamiento
canales   = [52,54]     # Canales a utilizar

#Cargar las estructuras
verdaderos_struct       = mat.loadmat("true_labels_al.mat")
entrenamiento_struct    = mat.loadmat("al.mat")
crudo_struct            = mat.loadmat("crudo_al.mat")

#Acceder a los valores verdaderos usano notacion de diccionario
verdaderos  = verdaderos_struct['true_y'][0,:]

#Clases para entrenamiento supervisado 
C1 = entrenamiento_struct['C1']
C2 = entrenamiento_struct['C2']

#Datos a clasificar
crudo       = crudo_struct['cnt']
cortes      = crudo_struct['mrk']['pos'][0][0][0]

#Diseño del filtro
b,a     = butter(4, [f_inf/(Fs/2),f_sup/(Fs/2)], btype='bandpass')

# #Probabilidades por clase
P_C1    = np.shape(C1)[2] / ( np.shape(C1)[2] + np.shape(C2)[2] )
P_C2    = np.shape(C2)[2] / ( np.shape(C1)[2] + np.shape(C2)[2] )

#Almacenar las varianzas
var_C1  = np.zeros(( len(canales), np.shape(C1)[2])  , dtype=float )
var_C2  = np.zeros(( len(canales), np.shape(C2)[2] ) , dtype=float )

#Aplicar filtro 1-D sobre la segunda dimension (indice 1)
C1  = lfilter(b,a,C1,1)
C2  = lfilter(b,a,C2,1)

#Calcular la varianza de cada canal en C1
for num_canal in range(len(canales)):

    #Calcular varianzas clase 1
    for num_exp in range(num_datos):
        var_C1[num_canal][num_exp] = np.var([C1[canales[num_canal]-1][k][num_exp] for k in range(750)])

    #Calcular varianzas clase 2
    for num_exp in range(num_datos):
        var_C2[num_canal][num_exp] = np.var([C2[canales[num_canal]-1][k][num_exp] for k in range(750)])


medias_C1   = np.zeros( len(var_C1) ,dtype=float )   # Guardar las medias de cada canal en C1
medias_C2   = np.zeros( len(var_C2) ,dtype=float )   # Guardar las medias de cada canal en C2
desv_C1     = np.zeros( len(var_C1) ,dtype=float )   # Guardar las desviaciones de cada canal en C1
desv_C2     = np.zeros( len(var_C2) ,dtype=float )   # Guardar las desviaciones de cada canal en C2

# Calcular media y desviacion estandar en cada canal
for i in range(len(var_C1)):
    medias_C1[i] = np.mean(var_C1[i][:])
    desv_C1[i]   = np.std (var_C1[i][:])

for i in range(len(var_C2)):
    medias_C2[i] = np.mean(var_C2[i][:])
    desv_C2[i]   = np.std (var_C2[i][:])

# Arreglo para procesar la senal cruda
varianzas = np.zeros((len(verdaderos),len(canales)),dtype=float)

# Recortar crudo y extraer varianza
for num_exp in range ( len(cortes) ):
    for num_canal in range ( len(canales) ):

        #Inicializacion de corte para no perderlo por salida del stack
        corte=[] 
        
        if num_exp<len(verdaderos)-1:
            corte = [crudo[i][canales[num_canal]-1] for i in range(cortes[num_exp]-1,cortes[num_exp+1])]
        else:
            corte = [crudo[i][canales[num_canal]-1] for i in range(cortes[num_exp]-1,len(crudo))]
        
        # Cambio de frecuencia de muestreo
        corte = [0.1*float(i) for i in corte]
        corte=resample_poly(corte,1,4)
            
        #Filtrar y calcular varianzas
        corte = lfilter(b,a,corte,0)
        varianzas[num_exp][num_canal] = np.var([corte[i] for i in range(int(t_exp*Fs)-1)])

c1 = 0          # Contar clasificados como C1
c2 = 0          # Contar clasificados como C2
aciertos = 0    # Contar aciertos 

# Almacenar la clasificacion
respuestas = np.zeros(len(verdaderos))

# Tarea de clasificacion
for num_exp in range(len(varianzas)):

    PN_C1 = P_C1  
    PN_C2 = P_C2

    # Calcular probabilidad de cada clase
    for canal in range(len(varianzas[0])):

        # Tomar cada punto y calcular p(canal(N) dado C1)
        PN_C1 = PN_C1 * norm.pdf(varianzas[num_exp][canal],medias_C1[canal],desv_C1[canal])
        
        #Tomar cada punto y calcular p(canal(N) dado C2)
        PN_C2 = PN_C2 * norm.pdf(varianzas[num_exp][canal],medias_C2[canal],desv_C2[canal])
    
    # Comparar la probabilidad de las clases y asignar la mas alta
    if PN_C2 > PN_C1:
        c2+=1
        respuestas[num_exp]=2
    else:
        c1+=1
        respuestas[num_exp]=1

    # Si la clasificacion coincide con la respuesta, contar un acierto
    if respuestas[num_exp] == verdaderos[num_exp]:
        aciertos+=1

# Desplegar resultados
print("Total C1:",c1)
print("Total C2:",c2)
print("Aciertos:",aciertos,"Porcentaje:",100*aciertos/len(verdaderos),"%")