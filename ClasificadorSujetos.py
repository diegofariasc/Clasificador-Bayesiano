# Clasificador bayesiano con 9 sujetos, mismo movimiento
# Programa de honores UDLAP
from numpy        import var, mean, std, shape, zeros
from scipy.signal import butter, lfilter, resample_poly
from scipy.io     import loadmat
from scipy.stats  import norm


# Parametros de los archivos
sujetoA_nombreArchivo = "./Sujetos/S1"    # Nombre de la base de datos del sujetoA
sujetoB_nombreArchivo = "./Sujetos/S2"    # Nombre de la base de datos del sujetoB

# Parametros de la clasificacion
canales             = [1,2,3]   # Canales a considerar
Fs                  = 250       # Frecuencia de muestreo
cantDatosEntrenar   = 109       # Cantidad de muestras a usar en el entrenamiento   
claseUtilizada      = 'C1'      # Clase a utilizar para distinguir entre sujetos
bandaInferiorFiltro = 11        # Banda inferior de frecuencias a filtrar
bandaSuperiorFiltro = 13        # Banda superior de frecuencias a filtrar


def main():

    datosEntrena = entrenamiento()    # Entrenar el algoritmo
    print(datosEntrena)


def entrenamiento():
    
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

    # Fijar la cantidad de datos de entrenamiento a usar si 
    # los experimentos que se poseen son insuficientes para cumplir la demanda del usuario
    n_experimentos = min(n_experimentosA, n_experimentosB, cantDatosEntrenar)

    # Calcular las varianzas mediante comprension de lista 
    # Primero se obtiene la lista de las n_muestras. Se calcula la varianza de las n_muestras
    # El proceso se repite para generar n_experimentos listas
    # Se repite nuevamente por cada canal en la lista dada 
    # Se concluye con una lista con shape (canales x varianzas)
    varianzas_SA = [[ var([sujetoA[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentos) ] for canal in canales]
    varianzas_SB = [[ var([sujetoB[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentos) ] for canal in canales]

    # Calcular por cada canal, la media y desviacion estandar
    medias_SA     = [mean(varianza) for varianza in varianzas_SA]   
    medias_SB     = [mean(varianza) for varianza in varianzas_SB]
    desviacion_SA = [std(varianza)  for varianza in varianzas_SA]
    desviacion_SB = [std(varianza)  for varianza in varianzas_SB]

    # Regresar diccionario con los datos de entrenamiento encontrados
    return {"medias_SA":medias_SA,"medias_SB":medias_SB,
            "desviacion_SA":desviacion_SA,"desviacion_SB":desviacion_SB}

main()

