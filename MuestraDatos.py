# Clasificador bayesiano con 9 sujetos, mismo movimiento
# Programa de honores UDLAP
from numpy              import var, mean, shape, zeros, asarray, median, std, percentile
from numpy.random       import randn 
from scipy.signal       import butter, lfilter, resample_poly
from scipy.io           import loadmat
from scipy.stats        import norm, kurtosis, skew, normaltest, shapiro
from random             import shuffle, seed
from matplotlib.pyplot  import plot, hist, show
from statsmodels.graphics.gofplots import qqplot
from math               import sqrt
from operator           import itemgetter

def mostrar():

    # Cargar la base de datos
    sujetoA = loadmat(sujetoA_nombreArchivo+".mat")[claseUtilizada]
    sujetoB = loadmat(sujetoB_nombreArchivo+".mat")[claseUtilizada]

    # Diseño del filtro
    b,a = butter(4, [bandaInferiorFiltro/(Fs/2),bandaSuperiorFiltro/(Fs/2)], btype='bandpass')

    # Filtrado de los datos
    sujetoA  = lfilter(b,a,sujetoA,1)     # Filtrar sujeto A. Clase 1. Dimensión 2 (indice 1)
    sujetoB  = lfilter(b,a,sujetoB,1)     # Filtrar sujeto B. Clase 1. Dimensión 2 (indice 1)
    
    # Desempaquetar dimensiones de la matriz
    n_muestras,n_experimentosA = shape(sujetoA)[1:3]
    n_experimentosB = shape(sujetoB)[2]   
    
    # Calcular las varianzas mediante comprension de lista 
    # Primero se obtiene la lista de las n_muestras. Se calcula la varianza de las n_muestras
    # El proceso se repite para generar n_experimentos listas
    # Se repite nuevamente por cada canal en la lista dada 
    # Se concluye con una lista con shape (canales x varianzas)
    varianzas_SA = [[ std([sujetoA[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentosA) ] for canal in canales]
    varianzas_SB = [[ std([sujetoB[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentosB) ] for canal in canales]

    desplazamiento=4

    # Desviacion estandar 
    #umbralCanalesA=[(mean(canal)-desplazamiento*std(canal),mean(canal)+desplazamiento*std(canal)) for canal in varianzas_SA]
    #umbralCanalesB=[(mean(canal)-desplazamiento*std(canal),mean(canal)+desplazamiento*std(canal))for canal in varianzas_SB]

    # Desviacion media absolta (MAD)
    umbralCanalesA=[(mean(canal)-desplazamiento*MAD(canal),mean(canal)+desplazamiento*MAD(canal)) for canal in varianzas_SA]
    umbralCanalesB=[(mean(canal)-desplazamiento*MAD(canal),mean(canal)+desplazamiento*MAD(canal))for canal in varianzas_SB]

    # Rango intercuartil
    #umbralCanalesA=[IQR(canal) for canal in varianzas_SA]
    #umbralCanalesB=[IQR(canal) for canal in varianzas_SB]

    # Filtrar los datos para remover outliers
    n_canalesA, n_canalesB = len(varianzas_SA), len(varianzas_SB)
    varianzas_SA_Filtradas = filtrar(n_experimentosA,n_canalesA,umbralCanalesA,varianzas_SA)
    varianzas_SB_Filtradas = filtrar(n_experimentosB,n_canalesB,umbralCanalesB,varianzas_SB)

    minimo_elementos = min(len(varianzas_SA_Filtradas[0]),len(varianzas_SB_Filtradas[0]))
    ajustarNumeroExperimentos( varianzas_SA_Filtradas, minimo_elementos)
    ajustarNumeroExperimentos( varianzas_SB_Filtradas, minimo_elementos)

    print("Datos restantes A: ",len(varianzas_SA_Filtradas[0]))
    print("Datos restantes B: ",len(varianzas_SB_Filtradas[0]))

    hist(varianzas_SB_Filtradas[2])
    show()

    # Ejemplo de valores que se sabe son gaussianos
    #varianzas_SA = [randn(159)*100 for i in range(3)] 
    #varianzas_SB = [randn(159)*100 for i in range(3)] 

    #qqplot(asarray(varianzas_SA[2]), line='s')
    #show()
    
    # Pruebas Shapiro-Wilks 
    print("-------- Pruebas Shapiro-Wilks --------")
    [print("Prueba SA Canal",i,": ",probarSiGaussiano(varianzas_SA_Filtradas[i],'Shapiro')) for i in range(len(varianzas_SA_Filtradas))] 
    [print("Prueba SB Canal",i,": ",probarSiGaussiano(varianzas_SB_Filtradas[i],'Shapiro')) for i in range(len(varianzas_SB_Filtradas))] 

    # Pruebas D’Agostino’s K^2
    print("---------- D’Agostino’s K^2 -----------")
    [print("Prueba SA Canal",i,": ",probarSiGaussiano(varianzas_SA_Filtradas[i],'D’Agostino')) for i in range(len(varianzas_SA_Filtradas))] 
    [print("Prueba SB Canal",i,": ",probarSiGaussiano(varianzas_SB_Filtradas[i],'D’Agostino')) for i in range(len(varianzas_SB_Filtradas))] 

    # Imprimir caracteristicas de los datos
    print("Caracteristicas de los datos ")
    [print("SA Canal",i," Kurtosis:",kurtosis(varianzas_SA[i]), " Asimetria:",skew(varianzas_SA[i])) for i in range(len(varianzas_SA))] 
    [print("SB Canal",i," Kurtosis:",kurtosis(varianzas_SB[i]), " Asimetria:",skew(varianzas_SB[i])) for i in range(len(varianzas_SB))] 

def probarSiGaussiano(datos, tipo, margen=0.05):

    # Hipotesis nula: los datos se ajustan a una distribucion normal

    estadistico, p = 0,0

    if (tipo == 'D’Agostino'):          
        estadistico, p  = normaltest(datos)     # D'Agostino
    elif (tipo == 'Shapiro'):
        estadistico, p  = shapiro(datos)        # Shapiro-Wilks
    else:
        raise ValueError                        # Prueba no definida

    if p > margen:
        return True     # No hay evidencia para rechazar hipotesis nula
    else:
        return [False,"Margen: "+str(margen), "P: "+str(p)]    # Rechazar hipotesis nula

# Median absolute distribution 
def MAD(datos):
    mediana = median(datos)
    return median([abs(x-mediana) for x in datos])
    
# Rango intercuartil
def IQR(datos):
    datos = sorted(datos)
    Q1,Q3 = percentile(datos,[25,75])
    IQR = Q3 - Q1
    return (Q1 - (1.5 * IQR),Q3 +(1.5 * IQR))

def filtrar(n_experimentos, n_canales, umbral, datos):

    # Generar lista para almacenar valores que pasen [ [], [], [] ]
    filtrado = [[] for i in range(n_canales)]

    # Iterar sobre los experimentos
    for exp in range(n_experimentos):

        # Revisar canales
        pasa = True
        for canal in range(n_canales):
            pasa = pasa and datos[canal][exp] > umbral[canal][0] and datos[canal][exp] < umbral[canal][1]

        # Si pasan, agregarlos al conjunto
        if pasa:
            for canal in range(n_canales):
                filtrado[canal].append(datos[canal][exp])

    return filtrado

def ajustarNumeroExperimentos(datos, cantidad):

    # Si no hay que borrar nada, solo regresar del metodo
    if cantidad >= len(datos[0]):
        return

    distanciasCanal = [[] for i in range(len(datos))]

    # Calcular |media-xi|
    for canal in range(len(datos)):

        media = median(datos[canal])
        distanciasCanal[canal]=[abs(media-dato) for dato in datos[canal]]

    # Para almacenar las relaciones de distancias con sus indices
    distanciasIndices=[]

    # Generar conjunto [ (distancia sumada, indice) ... ] 
    for exp in range(len(distanciasCanal[0])):
        suma = 0
        for canal in range(len(distanciasCanal)):
            suma = suma + distanciasCanal[canal][exp]

        distanciasIndices.append((suma,exp))
            
    # Ordenar por distancia
    distanciasIndices.sort(key=itemgetter(0))

    # Obtener una lista con la cantidad de elementos a borrar
    distanciasCanalNecesarias = distanciasIndices[0:len(datos[0])-cantidad]

    # Ordenar por indice
    distanciasCanalNecesarias.sort(key=itemgetter(1),reverse=True)

    # Borrar datos
    for elementoRemover in distanciasCanalNecesarias:
        for canal in datos:
            del canal[elementoRemover[1]]

# Parametros de los archivos
sujetoA_nombreArchivo = "./Sujetos/S1"   # Nombre de la base de datos del sujetoA
sujetoB_nombreArchivo = "./Sujetos/S2"    # Nombre de la base de datos del sujetoB

# Parametros de la clasificacion
canales             = [1,2,3]               # Canales a considerar
Fs                  = 250                   # Frecuencia de muestreo
porcentajeEntrenar  = 0.5                   # Porcentaje de los datos que se usaran para entrenar
iteraciones         = 30                    # Iteraciones a realizar
semillaKFold        = 1                     # Semilla para mezclar los datos 
claseUtilizada      = 'C1'                  # Clase a utilizar para distinguir entre sujetos
bandaInferiorFiltro = 3                     # Banda inferior de frecuencias a filtrar
bandaSuperiorFiltro = 9                     # Banda superior de frecuencias a filtrar

mostrar()