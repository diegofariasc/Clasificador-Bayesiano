# Pruebas para validar que los datos tengan una forma gaussiana
# Programa de honores UDLAP

from numpy              import var, mean, shape, zeros, asarray, median, std, percentile
from numpy.random       import randn 
from scipy.signal       import butter, lfilter, resample_poly
from scipy.io           import loadmat
from scipy.stats        import norm, kurtosis, skew, normaltest, shapiro
from random             import shuffle, seed, sample
from matplotlib.pyplot  import plot, hist, show
from statsmodels.graphics.gofplots import qqplot
from math               import sqrt, isnan
from operator           import itemgetter
from pandas             import read_excel


def realizarPruebas(sujetoA_nombreArchivo,sujetoB_nombreArchivo,claseUtilizada,
                    bandaInferiorFiltro,bandaSuperiorFiltro,canales,Fs,desplazamiento):

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

    # Desviacion estandar 
    #umbralCanalesA=[(mean(canal)-desplazamiento*std(canal),mean(canal)+desplazamiento*std(canal)) for canal in varianzas_SA]
    #umbralCanalesB=[(mean(canal)-desplazamiento*std(canal),mean(canal)+desplazamiento*std(canal))for canal in varianzas_SB]

    # Desviacion media absolta (MAD)
    #umbralCanalesA=[(median(canal)-desplazamiento*MAD(canal),median(canal)+desplazamiento*MAD(canal)) for canal in varianzas_SA]
    #umbralCanalesB=[(median(canal)-desplazamiento*MAD(canal),median(canal)+desplazamiento*MAD(canal))for canal in varianzas_SB]

    # Rango intercuartil
    umbralCanalesA=[IQR(canal,desplazamiento) for canal in varianzas_SA]
    umbralCanalesB=[IQR(canal,desplazamiento) for canal in varianzas_SB]

    # Filtrar los datos para remover outliers
    n_canalesA, n_canalesB = len(varianzas_SA), len(varianzas_SB)
    varianzas_SA_Filtradas = filtrar(n_experimentosA,n_canalesA,umbralCanalesA,varianzas_SA)
    varianzas_SB_Filtradas = filtrar(n_experimentosB,n_canalesB,umbralCanalesB,varianzas_SB)

    minimo_elementos = min(len(varianzas_SA_Filtradas[0]),len(varianzas_SB_Filtradas[0]))
    ajustarNumeroExperimentos( varianzas_SA_Filtradas, minimo_elementos)
    ajustarNumeroExperimentos( varianzas_SB_Filtradas, minimo_elementos)

    print("Datos restantes A: ",len(varianzas_SA_Filtradas[0]))
    print("Datos restantes B: ",len(varianzas_SB_Filtradas[0]))

    #hist(varianzas_SB_Filtradas[2])
    #show()

    #qqplot(asarray(varianzas_SA[2]), line='s')
    #show()
    
    # Pruebas Shapiro-Wilks 
    print("-------- Pruebas Shapiro-Wilks --------")
    PruebaShapiroA=[probarSiGaussiano(varianzas_SA_Filtradas[i],'Shapiro') for i in range(len(varianzas_SA_Filtradas))] 
    PruebaShapiroB=[probarSiGaussiano(varianzas_SB_Filtradas[i],'Shapiro') for i in range(len(varianzas_SB_Filtradas))] 

    # Pruebas D’Agostino’s K^2
    print("---------- D’Agostino’s K^2 -----------")
    PruebaAgostinoA=[probarSiGaussiano(varianzas_SA_Filtradas[i],'D’Agostino') for i in range(len(varianzas_SA_Filtradas))] 
    PruebaAgostinoB=[probarSiGaussiano(varianzas_SB_Filtradas[i],'D’Agostino') for i in range(len(varianzas_SB_Filtradas))] 

    # Devolver un diccionario con los resultados de las pruebas 
    # { Shapiro:  { SujetoA: (aprobados,fallados), SujetoB: (aprobados,fallados) }
    #   Agostino: { SujetoA: (aprobados,fallados), SujetoB: (aprobados,fallados) } }
    return {
        'Shapiro':  {
                        'SujetoA': (
                            len([prueba for prueba in PruebaShapiroA if prueba]),
                            len([prueba for prueba in PruebaShapiroA if not prueba])),
                        'SujetoB': (
                            len([prueba for prueba in PruebaShapiroB if prueba]),
                            len([prueba for prueba in PruebaShapiroB if not prueba]))
                    },
        'Agostino':  {
                        'SujetoA': (
                            len([prueba for prueba in PruebaAgostinoA if prueba]),
                            len([prueba for prueba in PruebaAgostinoA if not prueba])),
                        'SujetoB': (
                            len([prueba for prueba in PruebaAgostinoB if prueba]),
                            len([prueba for prueba in PruebaAgostinoB if not prueba]))
                    }
    }


def probarSiGaussiano(datos, tipo, margen=0.01):

    # Hipotesis nula: los datos se ajustan a una distribucion normal

    estadistico, p = 0,0

    if (tipo == 'D’Agostino'):          
        estadistico, p  = normaltest(datos)     # D'Agostino
    elif (tipo == 'Shapiro'):
        estadistico, p  = shapiro(datos)        # Shapiro-Wilks
    else:
        raise ValueError                        # Prueba no definida
    
    if p > margen:
        # No hay evidencia para rechazar hipotesis nula
        print("[Aprobada]")
        return True                             
    else:
        # Rechazar hipotesis nula
        print("[Fallada] Margen: "+str(margen), "P: "+str(p))
        return False

# Median absolute distribution 
def MAD(datos):
    mediana = median(datos)
    return median([abs(x-mediana) for x in datos])
    
# Rango intercuartil
def IQR(datos, desplazamiento=1.5):
    datos = sorted(datos)
    Q1,Q3 = percentile(datos,[25,75])
    IQR = Q3 - Q1
    return (Q1 - (desplazamiento * IQR),Q3 +(desplazamiento * IQR))

def filtrar(n_exp, n_canales, umbral, datos):

    # Generar lista para almacenar valores que pasen [ [], [], [] ]
    filtrado = [[] for i in range(n_canales)]
    
    # Iterar sobre los experimentos
    for exp in range(n_exp):
        
        # Revisar canales
        pasa = True
        for canal in range(n_canales):
            pasa = pasa and datos[canal][exp] > umbral[canal][0] and datos[canal][exp] < umbral[canal][1]

        # Si pasan, agregarlos al conjunto
        if pasa:
            for canal in range(n_canales):
                filtrado[canal].append(datos[canal][exp])

    return filtrado


def localizarIndicesConservar(n_exp, n_canales, umbral, datos):

    # Generar lista para almacenar indices de los valores que pasen [ ]
    indicesConservar = []

    # Iterar sobre los experimentos
    for exp in range(n_exp):

        # Revisar canales
        pasa = True
        for canal in range(n_canales):
            pasa = pasa and datos[canal][exp] > umbral[canal][0] and datos[canal][exp] < umbral[canal][1]

        # Si pasan, agregarlos al conjunto
        if pasa:
            indicesConservar.append( exp )

    return set ( indicesConservar )

def ajustarNumeroExperimentos(datos, cantidad):

    longitudCanal = min([len(canal) for canal in datos])

    # Si no hay que borrar nada, solo regresar del metodo
    if cantidad >= longitudCanal:
        return

    # Obtener lista de elementos a eliminar aleatoriamente
    seed(1)
    listaIndicesEliminar = sample(range(longitudCanal), longitudCanal-cantidad)

    # Orden en reversa para evitar indexOutOfBounds
    listaIndicesEliminar.sort(reverse=True)

    # Eliminar de la lista 
    [[canal.pop(indice) for indice in listaIndicesEliminar] for canal in datos]


def obtenerFrecuenciasCorteOptimas(archivo,hoja):

    # Cargar archivo 
    ArchivoEfectividades = read_excel(archivo,hoja)
    datosBandas = []

    # Iterar sobre las frecuencias de corte (filas)
    fila = 2
    while (fila<=23):

        # Obtener valores que no sean NaN
        efectividades = [valor for valor in ArchivoEfectividades.iloc[fila, 1:] if not isnan(valor)]

        # Empaquetado (Relacionar inferior y superior) [(C1_Inf,C1_Sup),(C2_Inf,C2_Sup),...]
        iterador = iter(efectividades)
        efectividades = list(zip(iterador, iterador))

        # Agregar datos al resultado
        datosBandas.append(efectividades)

        # Mover indice de la fila (Pandas)
        fila += 3

    return datosBandas

def main():

    frecuenciasCorteOptimas = obtenerFrecuenciasCorteOptimas('Efectividades.xlsx','K-Fold')

    casosShapiroAprobados   = 0
    casosShapiroFallados    = 0
    casosAgostinoAprobados  = 0
    casosAgostinoFallados   = 0  

    SujetoA = 0
    claseUtilizada = 'C2'

    for lineaBandas in frecuenciasCorteOptimas:

        SujetoA = SujetoA+1
        SujetoB = SujetoA

        for banda in lineaBandas:

            if claseUtilizada == 'C2':
                claseUtilizada = 'C1'
                SujetoB = SujetoB + 1
            else:
                claseUtilizada = 'C2'


            # Parametros de los archivos
            sujetoA_nombreArchivo = "./Sujetos/S"+str(SujetoA)  # Nombre de la base de datos del sujetoA
            sujetoB_nombreArchivo = "./Sujetos/S"+str(SujetoB)  # Nombre de la base de datos del sujetoB

            # Parametros
            desplazamiento          = 2.5
            canales                 = [1,2,3]   # Canales a considerar
            Fs                      = 250       # Frecuencia de muestreo
            
            bandaInferiorFiltro = banda[0]      # Banda inferior de frecuencias a filtrar
            bandaSuperiorFiltro = banda[1]      # Banda superior de frecuencias a filtrar

            resultados = realizarPruebas(sujetoA_nombreArchivo,sujetoB_nombreArchivo,claseUtilizada,
                                        bandaInferiorFiltro,bandaSuperiorFiltro,canales,Fs,desplazamiento)

            casosShapiroAprobados   += resultados['Shapiro'] ['SujetoA'][0] + resultados['Shapiro'] ['SujetoB'][0]
            casosShapiroFallados    += resultados['Shapiro'] ['SujetoA'][1] + resultados['Shapiro'] ['SujetoB'][1]
            casosAgostinoAprobados  += resultados['Agostino']['SujetoA'][0] + resultados['Agostino']['SujetoB'][0]
            casosAgostinoFallados   += resultados['Agostino']['SujetoA'][1] + resultados['Agostino']['SujetoB'][1]

    print("Desplazamiento:",desplazamiento)
    print('----------------------')  
    print("Shapiro aprovados: "+ str(casosShapiroAprobados))
    print("Shapiro fallados: "+ str(casosShapiroFallados))
    print("Porcentaje aprobacion shapiro: "+ str(100*casosShapiroAprobados/(casosShapiroAprobados+casosShapiroFallados)))
    print('----------------------')    
    print("Agostino aprovados: "+ str(casosAgostinoAprobados))
    print("Agostino fallados: "+ str(casosAgostinoFallados))
    print("Porcentaje aprobacion agostino: "+ str(100*casosAgostinoAprobados/(casosAgostinoAprobados+casosAgostinoFallados)))

