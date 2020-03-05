# Clasificador bayesiano con 9 sujetos, mismo movimiento
# Programa de honores UDLAP
from numpy              import var, mean, std, shape, zeros, asarray, median
from numpy.random       import randn 
from scipy.signal       import butter, lfilter, resample_poly
from scipy.io           import loadmat
from scipy.stats        import norm, kurtosis, skew, normaltest, shapiro
from random             import shuffle, seed
from matplotlib.pyplot  import plot, hist, show
from statsmodels.graphics.gofplots import qqplot

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

    hist(varianzas_SA[2])
    show()

    # Ejemplo de valores que se sabe son gaussianos
    #varianzas_SA = [randn(159)*100 for i in range(3)] 
    #varianzas_SB = [randn(159)*100 for i in range(3)] 

    qqplot(asarray(varianzas_SA[2]), line='s')
    show()
    
    # Pruebas Shapiro-Wilks 
    print("-------- Pruebas Shapiro-Wilks --------")
    [print("Prueba SA Canal",i,": ",probarSiGaussiano(varianzas_SA[i],'Shapiro')) for i in range(len(varianzas_SA))] 
    [print("Prueba SB Canal",i,": ",probarSiGaussiano(varianzas_SB[i],'Shapiro')) for i in range(len(varianzas_SB))] 

    # Pruebas D’Agostino’s K^2
    print("---------- D’Agostino’s K^2 -----------")
    [print("Prueba SA Canal",i,": ",probarSiGaussiano(varianzas_SA[i],'D’Agostino')) for i in range(len(varianzas_SA))] 
    [print("Prueba SB Canal",i,": ",probarSiGaussiano(varianzas_SB[i],'D’Agostino')) for i in range(len(varianzas_SB))] 

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