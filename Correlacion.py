# Identificacion de un sujeto mediante correlacion
# Programa de honores UDLAP

#from dit.other                      import renyi_entropy
from numpy                          import  mean, std, shape, array, median, var
from scipy.io                       import  loadmat
from scipy.signal                   import  butter, lfilter
from scipy.stats import norm
from scipy.stats.stats              import  pearsonr,kurtosis, skew
from dimFractal                     import  katz
from MuestraDatos                   import  ajustarNumeroExperimentos
from ClasificadorUnoTodosMultiBanda import  removerOutliersSujeto, \
                                            aplanarDimensionBandasFrecuenciaIndividual, \
                                            distribuirDatosEntrenamientoYClasificacion
from math                           import inf
import matplotlib.pyplot as plt

"""
El metodo carga los datos de un sujeto en la clase 
de movimiento especificada (ej. C1, C2,...)
"""
def cargar( numSujeto, claseUtilizada ):

    # Configurar nombre del archivo dependiendo del numero de sujeto dado
    nombreArchivo = "./Sujetos/S" + str(numSujeto) + ".mat"
    return loadmat(nombreArchivo)[claseUtilizada]


"""
El metodo aplica un filtro paso banda a los datos de un sujeto proporcionado
Input:  datosSujeto         array con los datos crudos del sujeto en la forma (canales, muestras, experimentos)
        frecuenciaInferior  valor entero con la frecuencia inferior del filtro paso banda
        frecuenciaSuperior  valor entero con la frecuencia superior del filtro paso banda
        Fs                  frecuencia de muestreo
"""
def filtrar ( datosSujeto, frecuenciaInferior, frecuenciaSuperior, Fs ):

    # Diseno del filtro
    b, a = butter(4, [frecuenciaInferior / (Fs / 2), frecuenciaSuperior / (Fs / 2)], btype='bandpass')
    return lfilter(b, a, datosSujeto, 1)


"""
El metodo aplica un filtro paso banda en distintas bandas de frecuencia
Input:  datosSujeto     array con los datos crudos del sujeto en la forma (canales, muestras, experimentos)
        bandas          lista de tuplas en la forma [ (fInf, fSup), (fInf, fSup), ... ]
        Fs              frecuencia de muestreo
Output: array en la forma (frecuencias, canales, muestras, experimentos)
"""
def filtrarMultibanda ( datosSujeto, bandas , Fs ):
    return array ( [ filtrar( datosSujeto, banda[0], banda[1], Fs ) for banda in bandas ] )


"""
El metodo recibe los datos filtrados de un sujeto, asi como una funcion 
caracteristica que se aplica a un conjunto determinado de datos 
Input:  datosSujeto     array en la forma (canales, muestras, experimentos)
        funcion         funcion a aplicar (mediante programacion funcional)
        canales         lista de canales en los que se debe operar 
Output: array en la forma canales, experimentos (caracteristicas)
Nota sobre rendimiento: En caso de descartar la dimension fractal como caracteristica
reemplazar el metodo de calculo de comprension de lista por una funcion preconstruida
en numpy para eficiencia
"""
def extraerCaracteristicas ( datosSujeto, funcion, canales ):

    return array( [ [funcion( [datosSujeto[canal - 1][j][k] for j in range( shape( datosSujeto )[1] ) ] )
                                                            for k in range( shape( datosSujeto )[2] ) ]
                                                            for canal in canales] )

"""
El metodo recibe los datos filtrados de un sujeto en multiples bandas de frecuencia.
Se encarga de extraer las caracteristicas por cada banda de frecuencia 
Input:  datosSujetoMultibanda   array en la forma (frecuencias, canales, muestras, experimentos)
        funcion                 funcion a aplicar (mediante programacion funcional)
        canales                 lista de canales en los que se debe operar 
Output: array en la forma ( frecuencias, canales, experimentos (caracteristicas) )
"""
def extraerCaracteristicasMultibanda ( datosSujetoMultibanda, funcion, canales ):
    return array( [ extraerCaracteristicas( frecuencia, funcion, canales) for frecuencia in datosSujetoMultibanda ] )


"""
El metodo normaliza el numero de experimentos de dados dos sujetos
al minimo entre ellos
Input:  sujetoA (canales,exp)
Output: Ninguno**
"""
def normalizarNumeroExperimentos ( sujetos ):

    # Calcular minimo
    minimo = min ( [ len( sujeto[0]) for sujeto in sujetos ] )

    # Invocar al metodo de reduccion para sujetoA y B
    [ ajustarNumeroExperimentos( sujeto, minimo) for sujeto in sujetos ]

def calcularAceptados (datos_C1, puntosMedios_C1, puntosDesviaciones_C1,
                       datos_C2, puntosMedios_C2, puntosDesviaciones_C2,
                       umbralCorrelacion, umbralKurtosis, umbralAsimetria,
                       verbose=False, umbralP1=0.76, umbralP2=0.76, umbralKurtosisC2=inf, umbralAsimetriaC2=inf):


    numCanales_C1, numExperimentos = shape (datos_C1 )
    numCanales_C2 = shape( datos_C2 ) [0]
    aceptados = 0

    for numExp in range ( numExperimentos ):

        experimento_C1 = []
        P_SA_C1 = 0.5

        experimento_C2 = []
        P_SA_C2 = 0.5

        for numCanal in range ( numCanales_C1 ):
            experimento_C1.append(datos_C1[numCanal][numExp])
            P_SA_C1 *= norm.pdf(datos_C1[numCanal - 1][numExp], puntosMedios_C1[numCanal - 1], puntosDesviaciones_C1[numCanal - 1])

        for numCanal in range ( numCanales_C2 ):
            experimento_C2.append(datos_C2[numCanal][numExp])
            P_SA_C2 *= norm.pdf(datos_C2[numCanal - 1][numExp], puntosMedios_C2[numCanal - 1], puntosDesviaciones_C2[numCanal - 1])

        diferenciaAsimetria_C1 = abs (skew( experimento_C1 ) - skew(puntosMedios_C1))
        diferenciakurtosis_C1 = abs (kurtosis( experimento_C1 ) - kurtosis(puntosMedios_C1))
        correlacion_C1 = pearsonr(puntosMedios_C1, experimento_C1)[0]

        diferenciaAsimetria_C2 = abs (skew( experimento_C2 ) - skew(puntosMedios_C2))
        diferenciakurtosis_C2 = abs (kurtosis( experimento_C2 ) - kurtosis(puntosMedios_C2))
        correlacion_C2 = pearsonr(puntosMedios_C2, experimento_C2)[0]

        pasaC1 = correlacion_C1 >= umbralCorrelacion and\
            diferenciaAsimetria_C1 <= umbralAsimetria and\
            diferenciakurtosis_C1 <= umbralKurtosis and\
            P_SA_C1 >= umbralP1
        pasaC2 = correlacion_C2 >= umbralCorrelacion and \
            diferenciaAsimetria_C2 <= umbralAsimetriaC2 and \
            diferenciakurtosis_C2 <= umbralKurtosisC2 and \
            P_SA_C2 >= umbralP2

        if pasaC1 and pasaC2:
            aceptados+=1


        if verbose:
            #print ( "Experimento %2i - C_C1: %.6f\tDK_C1: %.6f\tDA_C1: %.6f\tP_C1:%.6f\tC_C2: %.6f\tDK_C2: %.6f\tDA_C2: %.6f\tP_C2:%.6f\t%s\t%s" %
            print ( "Experimento %2i\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\t%s" %
                ( numExp, correlacion_C1, diferenciakurtosis_C1 , diferenciaAsimetria_C1, P_SA_C1,
                correlacion_C2, diferenciakurtosis_C2 , diferenciaAsimetria_C2, P_SA_C2, pasaC1, pasaC2) )

        #if numExp == 21:
        #    plt.plot ( experimento_C1, 'r')
        #    plt.plot(puntosMedios_C1, 'g')
        #    plt.plot ( experimento_C2, 'b')
        #    plt.plot(puntosMedios_C2, 'k')
        #    plt.title( "Experimento " + str(numExp) )
        #    plt.show()

    return aceptados



def realizarPruebaAutenticacion(sujetoA, sujetoB, canales, desplazamiento, Fs, porcentajeEntrenar,
                                frecuenciasFiltro_C1, frecuenciasFiltro_C2, funcionCaracteristica):

    # Cargar datos originales
    originalSujetoA_C1 = cargar( sujetoA, 'C1' )
    originalSujetoB_C1 = cargar( sujetoB, 'C1' )
    print( "Datos cargados C1\t- A:", shape( originalSujetoA_C1 ), "\t\tB:", shape( originalSujetoB_C1 ) )

    originalSujetoA_C2 = cargar( sujetoA, 'C2' )
    originalSujetoB_C2 = cargar( sujetoB, 'C2' )
    print( "Datos cargados C2\t- A:", shape( originalSujetoA_C2 ), "\t\tB:", shape( originalSujetoB_C2 ) )

    # Filtrado
    filtradoSujetoA_C1 = filtrarMultibanda(originalSujetoA_C1, frecuenciasFiltro_C1, Fs)
    filtradoSujetoB_C1 = filtrarMultibanda(originalSujetoB_C1, frecuenciasFiltro_C1, Fs)
    print("Datos filtrados C1\t- A:", shape( filtradoSujetoA_C1 ), "\tB:", shape( filtradoSujetoB_C1 ) )

    filtradoSujetoA_C2 = filtrarMultibanda(originalSujetoA_C2, frecuenciasFiltro_C2, Fs)
    filtradoSujetoB_C2 = filtrarMultibanda(originalSujetoB_C2, frecuenciasFiltro_C2, Fs)
    print("Datos filtrados C2\t- A:", shape( filtradoSujetoA_C2 ), "\tB:", shape( filtradoSujetoB_C2 ) )

    # Extraccion de caracteristicas
    caracteristicasSujetoA_C1 = extraerCaracteristicasMultibanda( filtradoSujetoA_C1, funcionCaracteristica, canales )
    caracteristicasSujetoB_C1 = extraerCaracteristicasMultibanda( filtradoSujetoB_C1, funcionCaracteristica, canales )
    print("\nExtraccion de caracteristicas C1:", funcionCaracteristica.__name__)
    print( "Datos tras extraer C1\t- A:", shape( caracteristicasSujetoA_C1 ), "\tB:", shape( caracteristicasSujetoB_C1 ) )

    caracteristicasSujetoA_C2 = extraerCaracteristicasMultibanda( filtradoSujetoA_C2, funcionCaracteristica, canales )
    caracteristicasSujetoB_C2 = extraerCaracteristicasMultibanda( filtradoSujetoB_C2, funcionCaracteristica, canales )
    print("Extraccion de caracteristicas C2:", funcionCaracteristica.__name__)
    print( "Datos tras extraer C2\t- A:", shape( caracteristicasSujetoA_C2 ), "\tB:", shape( caracteristicasSujetoB_C2 ) )

    # Remover outliers
    sinOutliersSujetoA_C1 = removerOutliersSujeto( caracteristicasSujetoA_C1, desplazamiento )
    sinOutliersSujetoB_C1 = removerOutliersSujeto( caracteristicasSujetoB_C1, desplazamiento )
    print( "\nDatos sin outliers C1\t- A:", shape( sinOutliersSujetoA_C1 ), "\tB:", shape( sinOutliersSujetoB_C1 ) )

    sinOutliersSujetoA_C2 = removerOutliersSujeto( caracteristicasSujetoA_C2, desplazamiento )
    sinOutliersSujetoB_C2 = removerOutliersSujeto( caracteristicasSujetoB_C2, desplazamiento )
    print( "Datos sin outliers C2\t- A:", shape( sinOutliersSujetoA_C2 ), "\tB:", shape( sinOutliersSujetoB_C2 ) )

    # Aplanamiento de datos
    aplanadoSujetoA_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliersSujetoA_C1 )
    aplanadoSujetoB_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliersSujetoB_C1 )
    print( "\nDatos tras aplanado C1\t- A:", shape( aplanadoSujetoA_C1 ), "\t\tB:", shape( aplanadoSujetoB_C1 ) )

    aplanadoSujetoA_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliersSujetoA_C2 )
    aplanadoSujetoB_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliersSujetoB_C2 )
    print( "Datos tras aplanado C2\t- A:", shape( aplanadoSujetoA_C2 ), "\t\tB:", shape( aplanadoSujetoB_C2 ) )

    # Normalizacion en el numero de experimentos
    normalizarNumeroExperimentos( [aplanadoSujetoA_C1, aplanadoSujetoB_C1, aplanadoSujetoA_C2, aplanadoSujetoB_C2] )
    print("\nDatos normalizados C1\t- A:", shape( aplanadoSujetoA_C1 ), "\t\tB:", shape( aplanadoSujetoB_C1 ))
    print("Datos normalizados C2\t- A:", shape( aplanadoSujetoA_C2 ), "\t\tB:", shape( aplanadoSujetoB_C2 ))

    # Distribuir datos en entrenamiento y clasificacion
    entrenamientoA_C1, clasificacionA_C1 = distribuirDatosEntrenamientoYClasificacion( aplanadoSujetoA_C1, porcentajeEntrenar )
    entrenamientoB_C1, clasificacionB_C1 = distribuirDatosEntrenamientoYClasificacion( aplanadoSujetoB_C1, porcentajeEntrenar )
    print("Clase 1")
    print("Distribucion datos A\t- Entrenar:", shape( entrenamientoA_C1 ), "\tClasificar:", shape( entrenamientoA_C1 ) )
    print("Distribucion datos B\t- Entrenar:", shape( entrenamientoB_C1 ), "\tClasificar:", shape( entrenamientoB_C1 ) )

    entrenamientoA_C2, clasificacionA_C2 = distribuirDatosEntrenamientoYClasificacion( aplanadoSujetoA_C2, porcentajeEntrenar )
    entrenamientoB_C2, clasificacionB_C2 = distribuirDatosEntrenamientoYClasificacion( aplanadoSujetoB_C2, porcentajeEntrenar )
    print("Clase 2")
    print("Distribucion datos A\t- Entrenar:", shape( entrenamientoA_C2 ), "\tClasificar:", shape( entrenamientoA_C2 ) )
    print("Distribucion datos B\t- Entrenar:", shape( entrenamientoB_C2 ), "\tClasificar:", shape( entrenamientoB_C2 ) )

    # Entrenar (vectores media)
    vectorMediasA_C1 = mean ( entrenamientoA_C1, axis=1 )
    vectorMediasB_C1 = mean ( entrenamientoB_C1, axis=1 )
    vectorDesviacionesA_C1 = std ( entrenamientoA_C1, axis=1 )
    vectorDesviacionesB_C1 = std ( entrenamientoB_C1, axis=1 )
    print("Vectores medias C1\t- A:", shape( vectorMediasA_C1 ), "\tB:", shape( vectorMediasB_C1 ) )
    print("Vectores desv. C1\t- A:", shape(vectorMediasA_C1), "\tB:", shape(vectorMediasB_C1))

    vectorMediasA_C2 = mean ( entrenamientoA_C2, axis=1 )
    vectorMediasB_C2 = mean ( entrenamientoB_C2, axis=1 )
    vectorDesviacionesA_C2 = std ( entrenamientoA_C2, axis=1 )
    vectorDesviacionesB_C2 = std ( entrenamientoB_C2, axis=1 )
    print("Vectores medias C2\t- A:", shape( vectorMediasA_C2 ), "\tB:", shape( vectorMediasB_C2 ) )
    print("Vectores desv. C2\t- A:", shape(vectorMediasA_C2), "\tB:", shape(vectorMediasB_C2))

    # Analizar aceptados
    aceptados = calcularAceptados( clasificacionA_C1, vectorMediasB_C1, vectorDesviacionesB_C1,
                                   clasificacionA_C2, vectorMediasB_C2, vectorDesviacionesB_C2, 0.72, inf, inf )
    total = shape(clasificacionB_C1)[1]
    rechazados = total-aceptados

    print ( "Aceptado: %i (%.2f%%)\tRechazados: %i (%.2f%%)" % ( aceptados, aceptados * 100 / total, rechazados, rechazados * 100 / total )   )


"""
Funcion principal
Evitar contaminacion del espacio global
"""
def main():

    # Parametros de la clasificacion
    sujetoA                 = 3             # Primer sujeto a comparar
    sujetoB                 = 7             # Segundo sujeto a comparar
    canales                 = [1,2,3]       # Canales a considerar
    desplazamiento          = 3.5           # Veces (MAD) distanciamiento de la mediana
    Fs                      = 250           # Frecuencia de muestreo
    porcentajeEntrenar      = 0.5           # Porcentaje de los datos que se usaran para entrenar
    funcionCaracteristica   = std           # Funcion para la extraccion de caracteristicas
    frecuenciasFiltro_C1    = [ (36, 43), (4, 8), (23, 35), (20,23), (33,49)  ]
    frecuenciasFiltro_C2    = [ (36, 43), (4, 8), (23, 35), (18,22), (34,45) ]

    # Invocar funcion de prueba
    realizarPruebaAutenticacion( sujetoA, sujetoB, canales, desplazamiento, Fs, porcentajeEntrenar,
                                 frecuenciasFiltro_C1, frecuenciasFiltro_C2, funcionCaracteristica )

