# Identificacion de un sujeto mediante correlacion
# con soporte para colecciones de multibandas de frecuencia de corte
# y vector de multiples caracteristicas
# Incluido: Modulo calculador de fronteras ideales para el rechazo de sujetos
# Programa de honores UDLAP

from entropy import *
from math                           import  inf
#from dit.other                      import  renyi_entropy
#from dit                            import  Distribution
from warnings                       import  filterwarnings
from mne.io                         import  read_raw_gdf
from mne                            import  events_from_annotations
from Correlacion                    import  filtrarMultibanda, extraerCaracteristicasMultibanda, \
                                            calcularAceptados, normalizarNumeroExperimentos
from numpy                          import  mean, std, shape, array, median, var, asarray, append, ndarray, \
                                            insert, vstack, reshape, transpose, fft, concatenate, equal, take_along_axis
from numpy.random                   import  rand
from scipy.io                       import  loadmat
from scipy.signal                   import  butter, lfilter
from scipy.stats                    import  norm, entropy, iqr
from scipy.stats.stats              import  pearsonr, spearmanr, kendalltau, kurtosis, skew
from dimFractal                     import  katz, hfd
from MuestraDatos                   import  ajustarNumeroExperimentos, MAD
from ClasificadorUnoTodosMultiBanda import  removerOutliersSujeto, \
                                            aplanarDimensionBandasFrecuenciaIndividual, \
                                            distribuirDatosEntrenamientoYClasificacion

def obtenerColor( color ):
    if color == 'HEADER'    : return '\033[95m'
    if color == 'BLUE'      : return '\033[94m'
    if color == 'OK'    : return '\033[93m'
    if color == 'YELLOW'    : return '\033[92m'
    if color == 'RED'       : return '\033[91m'
    if color == 'END'       : return '\033[0m'
    if color == 'BOLD'      : return '\033[1m'
    if color == 'UNDERLINE' : return '\033[4m'

def printC( color, mensaje ):
    print( obtenerColor( color ) + mensaje + obtenerColor('END') )

"""
El metodo carga un archivo GDF empleando la libreria mne
Input:  nombreArchivo = Nombre del GDF a cargar
Output: archivo crudo GDF cargado
"""
def cargarGDFcrudo( sujeto, sesion, ubicacion ):
    if sesion >= 4:
        nombreArchivo = ubicacion + "/B0" + str( sujeto ) + "0" + str( sesion ) + "E.gdf"
    else:
        nombreArchivo = ubicacion + "/B0" + str( sujeto ) + "0" + str( sesion ) + "T.gdf"
    return read_raw_gdf( nombreArchivo , preload=True, verbose='ERROR')

"""
El metodo invierte las llaves y valores de un diccionario
Input:  diccionario (dict)
Output: diccionario con llaves y valores revertidos 
"""
def invertirDiccionario( diccionario ):
    return { valor: llave for llave, valor in diccionario.items() }



"""
El metodo extrae los experimentos de una senal cruda continua
Nota: Se asume que los datos siguen la codificacion del dataset 2b
Nota: Se usa el diccionario de codificacion para mantener coherencia con
el documento anexo a la base de datos, aunque es innecesario. 
EL CODIGO SE PUEDE OPTIMIZAR 
"""
def extraerExperimentos ( senal, puntosCorte, codificacionCorte, tiempo, Fs = 250,
                          duracionTotalExperimento = 4.5, tomarDesdeCruz=False, duracionCruz = 3 ):
                          #,experimentosRechazar = set([]) ):

    # Calcular muestras a desplazar a la izquierda
    # El documento indica que la duracion de un experimento es 4.5 seg
    desplazamientoTiempo = int ( ( duracionTotalExperimento- tiempo ) / 2 ) * Fs

    # Indicador para rechazar proximo experimento
    rechazar    = False

    # Almacenar muestras recortadas
    muestras_C1 = []
    muestras_C2 = []

    #exp = 0

    # Recorrer puntos de corte
    for puntoCorte in puntosCorte:

        # Sustraer datos del punto de corte
        muestra, _, codigo = puntoCorte

        # En caso de encontrar un codigo de flecha a la izquierda
        if codificacionCorte[ codigo ] == '769':

            #if not (rechazar or exp in experimentosRechazar):
            if not rechazar:
                if tomarDesdeCruz:
                    muestra -= duracionCruz * Fs
                muestras_C1.append( senal[ : , muestra + desplazamientoTiempo : muestra + desplazamientoTiempo + int(tiempo * Fs) ] )
            rechazar = False
            #exp += 1

        # En caso de encontrar un codigo de flecha a la izquierda
        if codificacionCorte[ codigo ] == '770':
            if not rechazar:
            #if not (rechazar or exp in experimentosRechazar):
                if tomarDesdeCruz:
                    muestra -= duracionCruz * Fs
                muestras_C2.append( senal[ : , muestra + desplazamientoTiempo : muestra + desplazamientoTiempo + int(tiempo * Fs) ] )
            rechazar = False
            #exp += 1

        # En caso de encontrar un codigo de Reject
        # Notificar a la siguiente iteracion para evitar su agregacion
        elif codificacionCorte[ codigo ] == '1023':
           rechazar = True

    # Convertir a np arrays
    muestras_C1 = array(muestras_C1)
    muestras_C2 = array(muestras_C2)

    muestras_C1 = transpose(muestras_C1, (1, 2, 0))
    muestras_C2 = transpose(muestras_C2, (1, 2, 0))

    # Los datos estan en forma ( muestra, canal, experimento )
    # Transponer al orden convencional ( canal, muestra, experimento )
    return ( muestras_C1, muestras_C2 )


"""
El metodo permite la carga de multiples GDF del sujeto especificado 
en las sesiones indicadas
"""
def cargarGDFMultiplesSesiones( sujeto, sesiones, ubicacionDatos, verbose = False ):

    return [ cargarGDFcrudo( sujeto, sesion, ubicacionDatos) for sesion in sesiones ]

def extraerDatosCrudosGDFMultiplesSesiones( listaGDFs ):
    return [ GDF._data [:3,:] for GDF in listaGDFs ]

def extraerCodificacionesCorteMultiplesSesiones( listaGDFs ):
    return [ invertirDiccionario( events_from_annotations ( GDF, verbose='ERROR' )[1] ) for GDF in listaGDFs ]

def extraerPuntosCorteMultiplesSesiones( listaGDFs ):
    return [ events_from_annotations ( GDF, verbose='ERROR' )[0] for GDF in listaGDFs ]

def extraerExperimentosMultiplesSesiones(   listaDatosCrudos, listaPuntosCorte, listaCodificaciones, tiempoConsiderado,
                                            tomarDesdeCruz):
                                            #, listaDeRechazo=set([]) ):

    # Listas de retorno
    datosC1 = None
    datosC2 = None

    # Computar una lista de rechazo si no se dio
    #if len(listaDeRechazo) == 0:
    #    print("Crear set")
    #    listaDeRechazo = [ set([]) * len(listaDatosCrudos) ]
    #else:
    #    print("Lista interna:",listaDeRechazo)

    # Variable para indicar si los anteriores ya fueron inicializados
    datos_inicializados = False

    # Iterar en paralelo las listas con sus respectivos datos crudos,
    # puntos de corte y listas de codificacion
    #for datosCrudos, puntosCorte, codificacionCorte, rechazados in zip( listaDatosCrudos, listaPuntosCorte, listaCodificaciones, listaDeRechazo ):
    for datosCrudos, puntosCorte, codificacionCorte in zip( listaDatosCrudos, listaPuntosCorte, listaCodificaciones ):

        # Extraer experimentos (en C1 y C2) de la n-esima sesion
        C1, C2 = extraerExperimentos(datosCrudos, puntosCorte, codificacionCorte, tiempoConsiderado, tomarDesdeCruz=tomarDesdeCruz)
                                    #, experimentosRechazar = rechazados )


        # Ver si la lista de retorno ya esta inicializada.
        # Si es asi solo apilar los datos recien extraidos
        if ( datos_inicializados ):

            # Almacenamiento temporal en variable para evitar comportamientos inesperados
            nuevosDatosC1 = concatenate( ( datosC1, C1 ), -1 )
            nuevosDatosC2 = concatenate( ( datosC2, C2 ), -1 )
            datosC1 = nuevosDatosC1
            datosC2 = nuevosDatosC2

        # De lo contrario inicializar la lista y denotarlo en la variable
        else:
            datosC1 = C1
            datosC2 = C2
            datos_inicializados = True

    return ( datosC1, datosC2 )


def filtrarPorColeccionesMultibanda( datosCrudos, listaColeccionesBandas, Fs ):

    return [ filtrarMultibanda( datosCrudos, coleccionBandas, Fs) for coleccionBandas in listaColeccionesBandas ]

def extraerMultiplesCaracteristicasColeccionesMultibanda( datos, listaFuncionesCasracteristicas, canales ):

    return [ [  extraerCaracteristicasMultibanda( coleccionBandas, funcionCaracteristica, canales )
                for funcionCaracteristica in listaFuncionesCasracteristicas ]
                for coleccionBandas in datos ]

def ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda( datos, desplazamiento ):

    return [ [  removerOutliersSujeto( caracteristica, desplazamiento )
                for caracteristica in coleccionBandas ]
                for coleccionBandas in datos ]

def aplanarMultiplesCaracteristicasColeccionesMultibanda( datos ):

    return [ [  aplanarDimensionBandasFrecuenciaIndividual( caracteristica )
                for caracteristica in coleccionBandas ]
                for coleccionBandas in datos ]

def normalizarMultiplesCaracteristicasColeccionesMultibanda ( datos ):

    # Calcular minimo
    lista_minimos =[]
    [[[lista_minimos.append(shape(caracteristica)[1])   for caracteristica  in coleccionBandas ]
                                                        for coleccionBandas in lista ]
                                                        for lista           in datos ]
    minimo = min ( lista_minimos )

    # Invocar al metodo de reduccion para sujetoA y B
    [[[ ajustarNumeroExperimentos( caracteristica, minimo)  for caracteristica  in coleccionBandas]
                                                            for coleccionBandas in lista]
                                                            for lista in datos ]

def calcularParametrosEntrenamiento( datos, parametro, expEntrenar ):

    return [ [  parametro( shuffle_along_axis( array(caracteristica), 1 )[:,:expEntrenar], axis = 1 )
                for caracteristica in coleccionBandas ]
                for coleccionBandas in datos ]


def shuffle_along_axis(a, axis):
    idx = rand(*a.shape).argsort(axis=axis)
    return take_along_axis(a,idx,axis=axis)

"""
Canales x experimentos
Vector de 
"""

def determinarEstadisticosVector( experimento, vectorMedio, vectorStd, compensacion = 0):

    diferenciaAsimetria = abs ( skew( experimento ) - skew( vectorMedio ) )
    diferenciakurtosis  = abs ( kurtosis( experimento ) - skew( vectorMedio ) )
    correlacion         = pearsonr( vectorMedio , experimento )[0]

    probabilidad = 1
    for x, media, std in zip( experimento, vectorMedio, vectorStd ):
        probabilidad *= norm.pdf( x + (compensacion * std) , media, std )

    return ( correlacion, probabilidad, diferenciakurtosis, diferenciaAsimetria )


def calcularFronteraEspecifica( datos, vectorMedio, vectorStd, correlacionEstatica, anularKurtosisYAsimetria, compensacion = 1 ):

    correlacionMin         = inf
    probabilidadMin        = inf
    diferenciaKurtosisMax  = inf if anularKurtosisYAsimetria else -inf
    diferenciaAsimetriaMax = inf if anularKurtosisYAsimetria else -inf


    # Extraer estadisticos por cada exp
    for exp in range ( shape( datos )[1] ):

        correlacion, probabilidad, \
        diferenciaKurtosis, diferenciaAsimetria = determinarEstadisticosVector( array(datos)[:,exp], vectorMedio, vectorStd, compensacion )

        # Identificar si hay nuevos estadisticos menores o mayores
        correlacionMin          = correlacion if correlacion < correlacionMin else correlacionMin
        probabilidadMin         = probabilidad if probabilidad < probabilidadMin else probabilidadMin
        diferenciaKurtosisMax   = diferenciaKurtosis if diferenciaKurtosis > diferenciaAsimetriaMax else diferenciaKurtosisMax
        diferenciaAsimetriaMax  = diferenciaAsimetria if diferenciaAsimetria > diferenciaAsimetriaMax else diferenciaAsimetriaMax


    # Anular calculo si se opta por correlacion estatica
    if not correlacionEstatica == None:
        correlacionMin = correlacionEstatica


    # Compensar con dist entre x's y puntos medios

    return ( correlacionMin, probabilidadMin , diferenciaKurtosisMax, diferenciaAsimetriaMax )


def obtenerFronteras( datos, medias, desviaciones, correlacionEstatica=None, anularKurtosisYAsimetria = False ):

    valores = array ([ [ [ calcularFronteraEspecifica( caracteristica, mediaCaract, desvCaract, correlacionEstatica, anularKurtosisYAsimetria )
                            for caracteristica, mediaCaract, desvCaract in zip ( coleccion, mediaColeccion, desvClase ) ]
                            for coleccion, mediaColeccion, desvClase    in zip ( clase, mediaClase, desvClase )  ]
                            for clase, mediaClase, desvClase            in zip ( datos, medias, desviaciones ) ] )

    # Dividir fronteras segun valor (ej Kurtosis, Skew, etc.)
    return ( valores[:,:,:,0].tolist(),    # Correlacion
             valores[:,:,:,1].tolist(),    # Probabilidad
             valores[:,:,:,2].tolist(),    # Diferencia kurtosis
             valores[:,:,:,3].tolist() )   # Diferencia asimetria






def determinarSiVectorAceptado( experimento, vectorMedio, vectorStd, umbralCorrel,
                                umbralProbabilidad, umbralKurtosis, umbralSkew ):



    correlacion, probabilidad, diferenciakurtosis, diferenciaAsimetria = determinarEstadisticosVector( experimento,
                                                                                                       vectorMedio,
                                                                                                       vectorStd )
    #print("Correlacion:", correlacion, "\tFrontera:", umbralCorrel)
    #print("Probabilidad:", probabilidad, "\tFrontera:", umbralProbabilidad)
    #print("Diferencia kurtosis:", diferenciakurtosis, "\tFrontera:", umbralKurtosis)
    #print("Diferencia asimetria:", diferenciaAsimetria, "\tFrontera:", umbralSkew)

    if ( correlacion < umbralCorrel or
         probabilidad < umbralProbabilidad or
         diferenciakurtosis > umbralKurtosis or
         diferenciaAsimetria > umbralSkew ):
        return False

    return True


def determinarSiExperimentoAceptado( exp, datos, medias, desviaciones, umbralesCorrelacion,
                                    umbralesProbabilidad, umbralesKurtosis, umbralesSkew ):

    for clase in range(len(datos)):

        for coleccion in range(len(datos[clase])):

            for caracteristica in range(len(datos[clase][coleccion])):

                if not determinarSiVectorAceptado(  array(datos[clase][coleccion][caracteristica])[:, exp],
                                                    medias[clase][coleccion][caracteristica],
                                                    desviaciones[clase][coleccion][caracteristica],
                                                    umbralesCorrelacion[clase][coleccion][caracteristica],
                                                    umbralesProbabilidad[clase][coleccion][caracteristica],
                                                    umbralesKurtosis[clase][coleccion][caracteristica],
                                                    umbralesSkew[clase][coleccion][caracteristica] ):
                    return False

    return True

def calcularAceptados ( datos, medias, desviaciones, umbralesCorrelacion,
                        umbralesProbabilidad, umbralesKurtosis, umbralesSkew,
                        numExperimentos, verbose=False ):

    aceptados = 0
    for exp in range ( numExperimentos ):

        if determinarSiExperimentoAceptado( exp, datos, medias, desviaciones, umbralesCorrelacion,
                                            umbralesProbabilidad, umbralesKurtosis, umbralesSkew ):
            aceptados += 1

    return aceptados



"""
Metodo principal
Input:  None
Output: None
"""
def main():

    bandasEspecificasC1 =[
        [(39,40), (36,38), (41,47) ],
        [(28,34), (41,46), (11,12) ],
        [(36,41), (42,48), (32,34) ],
        [(33,38), (11,12), (39,41) ],
        [(18,19), (20,24), (25,48) ],
        [(5,6),   (21,25) ],
        [(4,7),   (36,37), (38,46) ],
        [(31,33), (34,36), (37,38) ],
        [(39,40), (41,47), (47,48) ]
    ]

    bandasEspecificasC2 =[
        [(39,49), (37,38), (41,43) ],
        [(11,12), (31,32), (41,42) ],
        [(36,41), (35,36), (41,45) ],
        [(4,6),   (11,12), (34,38) ],
        [(20,21), (22,25), (27,29) ],
        [(4,7),   (38,46) ],
        [(5,7),   (37,38), (38,45) ],
        [(32,33), (34,36), (37,38) ],
        [(3,6),   (39,40), (41,45) ]
    ]

    for sujetoA in range (1,10):
        for sujetoB in range(1,10):
            ubicacionDatos              = "BCICIV_2b"           # Nombre del directorio donde se localizan los datos de los sujetos
            sujetoA                     = sujetoA               # Primer sujeto a comparar [Sujeto de acceso]
            sujetoB                     = sujetoB               # Segundo sujeto a comparar [Sujeto de entrenamiento]
            sesionesEntrenar            = [2,3]                 # Sesiones empleadas en el entrenamiento del sistema
            sesionesClasificar          = [1]                   # Sesiones empleadas en el entrenamiento del sistema
            canales                     = [1,2,3]               # Canales a considerar
            desplazamiento              = 3.5                   # Veces (MAD) distanciamiento de la mediana
            Fs                          = 250                   # Frecuencia de muestreo
            funcionesCaracteristicas    = [std,MAD,iqr]         # Funciones para la extraccion de caracteristicas
            tiempoConsiderado           = 2                     # Segundos considerados por experimento
            tomarDesdeCruz              = False                 # Si se toma el tiempo desde la cruz de ajuste o desde la flecha
            verbose                     = True                  # Si se muestra el procedimiento
            anularKurtosisAsimetria     = True                  # Indicador para anular las fronteras de kurtosis y asimetria
            correlacionEstatica         = None                  # Correlacion de frontera.
                                                                # Si se define en None, el sistema la calcula automaticamente
                                                                # Extrayendo la menor al clasificar los datos
                                                                # de entrenamiento contra si
            experimentosEntrenar        = 1000



            coleccionesBandasC1         = [ [ (36, 43), (4, 8), (23, 35), (20, 23), (18, 22) ],
                                        bandasEspecificasC1[sujetoB-1] ]

            coleccionesBandasC2         = [ [ (36, 43), (4, 8), (23, 35), (20, 23), (18, 22) ],
                                        bandasEspecificasC2[sujetoB-1] ]



            # Suprimir alertas de la libreria MNE
            filterwarnings("ignore", category=Warning)

            if verbose:
                print ( "Cargando archivos GDF..." )

            # Cargar los datos de los dos sujetos en las sesiones de entrenamiento
            gdfs_SujetoA_SesionesA = cargarGDFMultiplesSesiones( sujetoA, sesionesEntrenar, ubicacionDatos )
            gdfs_SujetoB_SesionesA = cargarGDFMultiplesSesiones( sujetoB, sesionesEntrenar, ubicacionDatos )

            # Cargar los datos de los dos sujetos en las sesiones de clasificacion
            gdfs_SujetoA_SesionesB = cargarGDFMultiplesSesiones( sujetoA, sesionesClasificar, ubicacionDatos )
            gdfs_SujetoB_SesionesB = cargarGDFMultiplesSesiones( sujetoB, sesionesClasificar, ubicacionDatos )

            if verbose:
                print( "Carga completa" )

            # Extraer datos de los GDF
            # Extraer primeros 3 canales (los demas son de potenciales visuales evocados P300)
            datosCrudosSujetoA_SesionesA = extraerDatosCrudosGDFMultiplesSesiones( gdfs_SujetoA_SesionesA )
            datosCrudosSujetoB_SesionesA = extraerDatosCrudosGDFMultiplesSesiones( gdfs_SujetoB_SesionesA )
            datosCrudosSujetoA_SesionesB = extraerDatosCrudosGDFMultiplesSesiones( gdfs_SujetoA_SesionesB )
            datosCrudosSujetoB_SesionesB = extraerDatosCrudosGDFMultiplesSesiones( gdfs_SujetoB_SesionesB )

            if verbose:
                printC( 'BLUE', "\nDatos cargados de los primeros" )
                shapeDatosSujetoA_SesionesA = [ shape(datos) for datos in datosCrudosSujetoA_SesionesA ]
                shapeDatosSujetoB_SesionesA = [ shape(datos) for datos in datosCrudosSujetoB_SesionesA ]
                shapeDatosSujetoA_SesionesB = [ shape(datos) for datos in datosCrudosSujetoA_SesionesB ]
                shapeDatosSujetoB_SesionesB = [ shape(datos) for datos in datosCrudosSujetoB_SesionesB ]

                print ( "Sesiones A.\tSujeto A:\t", shapeDatosSujetoA_SesionesA,"\tSujeto B:\t", shapeDatosSujetoB_SesionesA )
                print ( "Sesiones B.\tSujeto A:\t", shapeDatosSujetoA_SesionesB,"\tSujeto B:\t", shapeDatosSujetoB_SesionesB )


                printC('BLUE', "\nExtraccion de los puntos de corte")

            # Extraer el esquema de codificacion de los puntos de corte
            # Se extrae del primer elemento, aunque cualquier elemento es util, ya que este es constante
            # en todos los sujetos
            codificacionCortes_SujetoA_SesionesA = extraerCodificacionesCorteMultiplesSesiones( gdfs_SujetoA_SesionesA )
            codificacionCortes_SujetoB_SesionesA = extraerCodificacionesCorteMultiplesSesiones( gdfs_SujetoB_SesionesA )
            codificacionCortes_SujetoA_SesionesB = extraerCodificacionesCorteMultiplesSesiones( gdfs_SujetoA_SesionesB )
            codificacionCortes_SujetoB_SesionesB = extraerCodificacionesCorteMultiplesSesiones( gdfs_SujetoB_SesionesB )


            if verbose:
                printC( 'BOLD', "Codificacion de puntos de corte:" )
                print ("Sujeto A. Sesiones A" )
                print( codificacionCortes_SujetoA_SesionesA )
                print ("Sujeto B. Sesiones A" )
                print( codificacionCortes_SujetoB_SesionesA )
                print ("Sujeto A. Sesiones B" )
                print( codificacionCortes_SujetoA_SesionesB )
                print ("Sujeto B. Sesiones B" )
                print( codificacionCortes_SujetoB_SesionesB )

            # Extraer listas con puntos de corte
            puntosCorte_SujetoA_SesionesA = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoA_SesionesA ) #
            puntosCorte_SujetoB_SesionesA = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoB_SesionesA ) #
            puntosCorte_SujetoA_SesionesB = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoA_SesionesB ) #
            puntosCorte_SujetoB_SesionesB = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoB_SesionesB ) #

            # Cortar senal
            # En este punto se pierden las listas de multi-sesion y se converge en
            # la forma (canales, muestras, exp)
            datosC1_SujetoA_SesionesA, datosC2_SujetoA_SesionesA = extraerExperimentosMultiplesSesiones( datosCrudosSujetoA_SesionesA,
                                                                                                        puntosCorte_SujetoA_SesionesA,
                                                                                                        codificacionCortes_SujetoA_SesionesA,
                                                                                                        tiempoConsiderado,
                                                                                                        tomarDesdeCruz=tomarDesdeCruz )

            datosC1_SujetoB_SesionesA, datosC2_SujetoB_SesionesA = extraerExperimentosMultiplesSesiones( datosCrudosSujetoB_SesionesA,
                                                                                                        puntosCorte_SujetoB_SesionesA,
                                                                                                        codificacionCortes_SujetoB_SesionesA,
                                                                                                        tiempoConsiderado,
                                                                                                        tomarDesdeCruz=tomarDesdeCruz )

            datosC1_SujetoA_SesionesB, datosC2_SujetoA_SesionesB = extraerExperimentosMultiplesSesiones( datosCrudosSujetoA_SesionesB,
                                                                                                        puntosCorte_SujetoA_SesionesB,
                                                                                                        codificacionCortes_SujetoA_SesionesB,
                                                                                                        tiempoConsiderado,
                                                                                                        tomarDesdeCruz=tomarDesdeCruz)

            datosC1_SujetoB_SesionesB, datosC2_SujetoB_SesionesB = extraerExperimentosMultiplesSesiones( datosCrudosSujetoB_SesionesB,
                                                                                                        puntosCorte_SujetoB_SesionesB,
                                                                                                        codificacionCortes_SujetoB_SesionesB,
                                                                                                        tiempoConsiderado,
                                                                                                        tomarDesdeCruz=tomarDesdeCruz)

            # Mostrar cortes de senal
            if verbose:
                printC('BLUE', "\nResultado de recorte de senal")
                print ( "C1: Sesion A. Sujeto A", shape( datosC1_SujetoA_SesionesA ), "\tSesion A. Sujeto B", shape( datosC1_SujetoB_SesionesA ) )
                print ( "C1: Sesion B. Sujeto A", shape( datosC1_SujetoA_SesionesB ), "\tSesion B. Sujeto B", shape( datosC1_SujetoB_SesionesB ) )
                print ( "C2: Sesion A. Sujeto A", shape( datosC2_SujetoA_SesionesA ), "\tSesion A. Sujeto B", shape( datosC2_SujetoB_SesionesA ) )
                print ( "C2: Sesion B. Sujeto A", shape( datosC2_SujetoA_SesionesB ), "\tSesion B. Sujeto B", shape( datosC2_SujetoB_SesionesB ) )


            # Filtrado
            # En este punto, nuevamente se agrega una dimension adicional en el que se
            # contendran los bloques de diferentes bandas de frecuencia
            if verbose:
                printC('BLUE', "\nFiltrado de datos")

            filtrado_SujetoA_SesionesA_C1 = filtrarPorColeccionesMultibanda( datosC1_SujetoA_SesionesA, coleccionesBandasC1, Fs)
            filtrado_SujetoB_SesionesA_C1 = filtrarPorColeccionesMultibanda( datosC1_SujetoB_SesionesA, coleccionesBandasC1, Fs)
            filtrado_SujetoA_SesionesB_C1 = filtrarPorColeccionesMultibanda( datosC1_SujetoA_SesionesB, coleccionesBandasC1, Fs)
            filtrado_SujetoB_SesionesB_C1 = filtrarPorColeccionesMultibanda( datosC1_SujetoB_SesionesB, coleccionesBandasC1, Fs)

            if verbose:
                shapeFiltrado_SujetoA_SesionesA_C1 = [ shape(datos) for datos in filtrado_SujetoA_SesionesA_C1 ]
                shapeFiltrado_SujetoB_SesionesA_C1 = [ shape(datos) for datos in filtrado_SujetoB_SesionesA_C1 ]
                shapeFiltrado_SujetoA_SesionesB_C1 = [ shape(datos) for datos in filtrado_SujetoA_SesionesB_C1 ]
                shapeFiltrado_SujetoB_SesionesB_C1 = [ shape(datos) for datos in filtrado_SujetoB_SesionesB_C1 ]
                print("Datos filtrados C1\t Sesion A- A:", shapeFiltrado_SujetoA_SesionesA_C1, "\tB:", shapeFiltrado_SujetoB_SesionesA_C1 )
                print("Datos filtrados C1\t Sesion B- A:", shapeFiltrado_SujetoA_SesionesB_C1, "\tB:", shapeFiltrado_SujetoB_SesionesB_C1 )

            filtrado_SujetoA_SesionesA_C2 = filtrarPorColeccionesMultibanda( datosC2_SujetoA_SesionesA, coleccionesBandasC2, Fs)
            filtrado_SujetoB_SesionesA_C2 = filtrarPorColeccionesMultibanda( datosC2_SujetoB_SesionesA, coleccionesBandasC2, Fs)
            filtrado_SujetoA_SesionesB_C2 = filtrarPorColeccionesMultibanda( datosC2_SujetoA_SesionesB, coleccionesBandasC2, Fs)
            filtrado_SujetoB_SesionesB_C2 = filtrarPorColeccionesMultibanda( datosC2_SujetoB_SesionesB, coleccionesBandasC2, Fs)

            if verbose:
                shapeFiltrado_SujetoA_SesionesA_C2 = [ shape(datos) for datos in filtrado_SujetoA_SesionesA_C2 ] #
                shapeFiltrado_SujetoB_SesionesA_C2 = [ shape(datos) for datos in filtrado_SujetoB_SesionesA_C2 ] #
                shapeFiltrado_SujetoA_SesionesB_C2 = [ shape(datos) for datos in filtrado_SujetoA_SesionesB_C2 ] #
                shapeFiltrado_SujetoB_SesionesB_C2 = [ shape(datos) for datos in filtrado_SujetoB_SesionesB_C2 ] #
                print("Datos filtrados C2\t Sesion A- A:", shapeFiltrado_SujetoA_SesionesA_C2, "\tB:", shapeFiltrado_SujetoB_SesionesA_C2 ) #
                print("Datos filtrados C2\t Sesion B- A:", shapeFiltrado_SujetoA_SesionesB_C2, "\tB:", shapeFiltrado_SujetoB_SesionesB_C2 ) #

            # Extraccion de caracteristicas

            if verbose:
                printC('BLUE', "\nExtraccion de caracteristicas")

            caracteristicas_SujetoA_SesionesA_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionesA_C1, funcionesCaracteristicas, canales )
            caracteristicas_SujetoB_SesionesA_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesA_C1, funcionesCaracteristicas, canales )
            caracteristicas_SujetoA_SesionesB_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionesB_C1, funcionesCaracteristicas, canales )
            caracteristicas_SujetoB_SesionesB_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesB_C1, funcionesCaracteristicas, canales )

            if verbose:
                nombreFunciones = [ funcion.__name__ for funcion in funcionesCaracteristicas ]
                printC('BOLD', "Extraccion para C1. Funciones: " + str ( nombreFunciones ) )
                shapeCaracteristicas_SujetoA_SesionesA_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoA_SesionesA_C1 ]
                shapeCaracteristicas_SujetoB_SesionesA_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoB_SesionesA_C1 ]
                shapeCaracteristicas_SujetoA_SesionesB_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoA_SesionesB_C1 ]
                shapeCaracteristicas_SujetoB_SesionesB_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoB_SesionesB_C1 ]
                print ("Caracteristicas C1. Sesion A-\tA:", shapeCaracteristicas_SujetoA_SesionesA_C1,
                                                    "\tB:", shapeCaracteristicas_SujetoB_SesionesA_C1 )
                print ("Caracteristicas C1. Sesion B-\tA:", shapeCaracteristicas_SujetoA_SesionesB_C1,
                                                    "\tB:", shapeCaracteristicas_SujetoB_SesionesB_C1 )


            caracteristicas_SujetoA_SesionesA_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionesA_C2, funcionesCaracteristicas, canales ) #
            caracteristicas_SujetoB_SesionesA_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesA_C2, funcionesCaracteristicas, canales ) #
            caracteristicas_SujetoA_SesionesB_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionesB_C2, funcionesCaracteristicas, canales ) #
            caracteristicas_SujetoB_SesionesB_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesB_C2, funcionesCaracteristicas, canales ) #


            if verbose:
                shapeCaracteristicas_SujetoA_SesionesA_C2 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoA_SesionesA_C2 ] #
                shapeCaracteristicas_SujetoB_SesionesA_C2 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoB_SesionesA_C2 ] #
                shapeCaracteristicas_SujetoA_SesionesB_C2 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoA_SesionesB_C2 ] #
                shapeCaracteristicas_SujetoB_SesionesB_C2 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                        for coleccionBandas in caracteristicas_SujetoB_SesionesB_C2 ] #
                print ("Caracteristicas C2. Sesion A-\tA:", shapeCaracteristicas_SujetoA_SesionesA_C2,
                                                    "\tB:", shapeCaracteristicas_SujetoB_SesionesA_C2 )
                print ("Caracteristicas C2. Sesion B-\tA:", shapeCaracteristicas_SujetoA_SesionesB_C2,
                                                    "\tB:", shapeCaracteristicas_SujetoB_SesionesB_C2 )

            # Remocion de outliers
            if verbose:
                printC('BLUE', "\nRemocion de outliers")
                printC('BOLD', "Removedor de outliers para C1. MAD: " + str( desplazamiento ) )

            sinOutliers_SujetoA_SesionesA_C1 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionesA_C1, desplazamiento) #
            sinOutliers_SujetoB_SesionesA_C1 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesA_C1, desplazamiento) #
            sinOutliers_SujetoA_SesionesB_C1 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionesB_C1, desplazamiento) #
            sinOutliers_SujetoB_SesionesB_C1 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesB_C1, desplazamiento) #

            if verbose:

                shapeSinOutliers_SujetoA_SesionesA_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                    for coleccionBandas in sinOutliers_SujetoA_SesionesA_C1 ] #
                shapeSinOutliers_SujetoB_SesionesA_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                    for coleccionBandas in sinOutliers_SujetoB_SesionesA_C1 ] #
                shapeSinOutliers_SujetoA_SesionesB_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                    for coleccionBandas in sinOutliers_SujetoA_SesionesB_C1 ] #
                shapeSinOutliers_SujetoB_SesionesB_C1 = [ [ shape(caracteristica)   for caracteristica  in coleccionBandas ]
                                                                                    for coleccionBandas in sinOutliers_SujetoB_SesionesB_C1 ] #

                print ("Datos tras remocion C1. Sesion A-\tA:", shapeSinOutliers_SujetoA_SesionesA_C1,
                                                        "\tB:", shapeSinOutliers_SujetoB_SesionesA_C1 )
                print ("Datos tras remocion C1. Sesion B-\tA:", shapeSinOutliers_SujetoA_SesionesB_C1,
                                                        "\tB:", shapeSinOutliers_SujetoB_SesionesB_C1 )

                printC('BOLD', "Removedor de outliers para C2. MAD: " + str( desplazamiento ) )

            sinOutliers_SujetoA_SesionesA_C2 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionesA_C2, desplazamiento)
            sinOutliers_SujetoB_SesionesA_C2 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesA_C2, desplazamiento)
            sinOutliers_SujetoA_SesionesB_C2 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionesB_C2, desplazamiento)
            sinOutliers_SujetoB_SesionesB_C2 = ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesB_C2, desplazamiento)

            if verbose:
                shapeSinOutliers_SujetoA_SesionesA_C2 = [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                                for coleccionBandas in sinOutliers_SujetoA_SesionesA_C2]
                shapeSinOutliers_SujetoB_SesionesA_C2 = [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                                for coleccionBandas in sinOutliers_SujetoB_SesionesA_C2]
                shapeSinOutliers_SujetoA_SesionesB_C2 = [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                                for coleccionBandas in sinOutliers_SujetoA_SesionesB_C2]
                shapeSinOutliers_SujetoB_SesionesB_C2 = [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                                for coleccionBandas in sinOutliers_SujetoB_SesionesB_C2]

                print("Datos tras remocion C2. Sesion A-\tA:", shapeSinOutliers_SujetoA_SesionesA_C2,
                    "\tB:", shapeSinOutliers_SujetoB_SesionesA_C2)
                print("Datos tras remocion C2. Sesion B-\tA:", shapeSinOutliers_SujetoA_SesionesB_C2,
                    "\tB:", shapeSinOutliers_SujetoB_SesionesB_C2)

            # Aplanamiento de datos sin outliers
                printC('BLUE', "\nAplanamiento de datos")
                printC('BOLD', "Aplanamiento para datos de C1" )

            aplanado_SujetoA_SesionesA_C1 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoA_SesionesA_C1 )
            aplanado_SujetoB_SesionesA_C1 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoB_SesionesA_C1 )
            aplanado_SujetoA_SesionesB_C1 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoA_SesionesB_C1 )
            aplanado_SujetoB_SesionesB_C1 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoB_SesionesB_C1 )

            if verbose:

                shapeAplanado_SujetoA_SesionA_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesA_C1 ]
                shapeAplanado_SujetoB_SesionA_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesA_C1 ]
                shapeAplanado_SujetoA_SesionB_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesB_C1 ]
                shapeAplanado_SujetoB_SesionB_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C1 ]

                print ("Datos aplanados C1. Sesion A-\tA:", shapeAplanado_SujetoA_SesionA_C1 ,
                                                    "\tB:", shapeAplanado_SujetoB_SesionA_C1 )
                print ("Datos aplanados C1. Sesion B-\tA:", shapeAplanado_SujetoA_SesionB_C1 ,
                                                    "\tB:", shapeAplanado_SujetoB_SesionB_C1 )

                printC('BOLD', "Aplanamiento para datos de C2" )

            aplanado_SujetoA_SesionesA_C2 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoA_SesionesA_C2 )
            aplanado_SujetoB_SesionesA_C2 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoB_SesionesA_C2 )
            aplanado_SujetoA_SesionesB_C2 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoA_SesionesB_C2 )
            aplanado_SujetoB_SesionesB_C2 = aplanarMultiplesCaracteristicasColeccionesMultibanda( sinOutliers_SujetoB_SesionesB_C2 )

            if verbose:

                shapeAplanado_SujetoA_SesionA_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesA_C2 ]
                shapeAplanado_SujetoB_SesionA_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesA_C2 ]
                shapeAplanado_SujetoA_SesionB_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesB_C2 ]
                shapeAplanado_SujetoB_SesionB_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C2 ]

                print ("Datos aplanados C2. Sesion A-\tA:", shapeAplanado_SujetoA_SesionA_C2 ,
                                                    "\tB:", shapeAplanado_SujetoB_SesionA_C2 )
                print ("Datos aplanados C2. Sesion B-\tA:", shapeAplanado_SujetoA_SesionB_C2 ,
                                                    "\tB:", shapeAplanado_SujetoB_SesionB_C2 )

            # Normalizacion del numero de experimentos
                printC('BLUE', "\nNormalizacion del numero de datos")
                print("Misma cantidad entre sujeto A y B y en clase C1 y C2 [Discriminacion con ambos]")

            normalizarMultiplesCaracteristicasColeccionesMultibanda( [  aplanado_SujetoA_SesionesA_C1, aplanado_SujetoB_SesionesA_C1,
                                                                        aplanado_SujetoA_SesionesA_C2, aplanado_SujetoB_SesionesA_C2 ] )

            normalizarMultiplesCaracteristicasColeccionesMultibanda( [  aplanado_SujetoA_SesionesB_C1, aplanado_SujetoB_SesionesB_C1,
                                                                        aplanado_SujetoA_SesionesB_C2, aplanado_SujetoB_SesionesB_C2 ] )


            if verbose:

                shapeAplanado_SujetoA_SesionA_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesA_C1 ]
                shapeAplanado_SujetoB_SesionA_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesA_C1 ]
                shapeAplanado_SujetoA_SesionB_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesB_C1 ]
                shapeAplanado_SujetoB_SesionB_C1 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C1 ]

                shapeAplanado_SujetoA_SesionA_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesA_C2 ]
                shapeAplanado_SujetoB_SesionA_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesA_C2 ]
                shapeAplanado_SujetoA_SesionB_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionesB_C2 ]
                shapeAplanado_SujetoB_SesionB_C2 = [ [ shape(caracteristica) for caracteristica  in coleccionBandas ]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C2 ]


                print("Datos normalizados C1. Sesion A-\tA:", shapeAplanado_SujetoA_SesionA_C1,
                                                    "\tB:", shapeAplanado_SujetoB_SesionA_C1 )
                print("Datos normalizados C1. Sesion B-\tA:", shapeAplanado_SujetoA_SesionB_C1,
                                                    "\tB:", shapeAplanado_SujetoB_SesionB_C1 )
                print("Datos normalizados C2. Sesion A-\tA:", shapeAplanado_SujetoA_SesionA_C2,
                                                    "\tB:", shapeAplanado_SujetoB_SesionA_C2 )
                print("Datos normalizados C2. Sesion B-\tA:", shapeAplanado_SujetoA_SesionB_C2,
                                                    "\tB:", shapeAplanado_SujetoB_SesionB_C2 )


            # Entrenar

            if verbose:
                printC('BLUE', "\nEntrenamiento. Empleando sesion A")

            vectoresMedias_C1       = calcularParametrosEntrenamiento( aplanado_SujetoB_SesionesA_C1, mean, experimentosEntrenar )
            vectoresDesviaciones_C1 = calcularParametrosEntrenamiento( aplanado_SujetoB_SesionesA_C1, std, experimentosEntrenar )
            vectoresMedias_C2       = calcularParametrosEntrenamiento( aplanado_SujetoB_SesionesA_C2, mean, experimentosEntrenar )
            vectoresDesviaciones_C2 = calcularParametrosEntrenamiento( aplanado_SujetoB_SesionesA_C2, std, experimentosEntrenar )

            if verbose:
            # Mostrar dimensiones de los vectores de entrenamiento en C1

                shapeVectoresMedias_C1 = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresMedias_C1]
                shapeDesviaciones_C1   = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresDesviaciones_C1]
                shapeVectoresMedias_C2 = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresMedias_C2]
                shapeDesviaciones_C2 = [[shape(caracteristica)      for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresDesviaciones_C2]

                print("Vectores medios C1\t- B:", shapeVectoresMedias_C1 )
                print("Vectores desviacion C1\t- B:", shapeDesviaciones_C1 )
                print("Vectores medios C2\t- B:", shapeVectoresMedias_C2)
                print("Vectores desviacion C2\t- B:", shapeDesviaciones_C2)

            # Identificador de fronteras
            fronteraCorrelacion, fronteraProbabilidad, \
            fronteraKurtosis, fronteraSkew = obtenerFronteras(  [ aplanado_SujetoB_SesionesA_C1, aplanado_SujetoB_SesionesA_C2 ],
                                                                [ vectoresMedias_C1, vectoresMedias_C2 ],
                                                                [ vectoresDesviaciones_C1, vectoresDesviaciones_C2 ], correlacionEstatica,
                                                                anularKurtosisAsimetria)
            if verbose: 
                print("A_B_C1:",array(aplanado_SujetoA_SesionesB_C1[0][0])[:, 0], "A_B_C2:",array(aplanado_SujetoA_SesionesB_C2[0][0])[:, 0])
                print("B_B_C1:",array(aplanado_SujetoB_SesionesA_C1[0][0])[:,0], "B_B_C2:",array(aplanado_SujetoB_SesionesA_C2[0][0])[:,0])
                print(vectoresMedias_C1)
                print(vectoresMedias_C2)
                print(vectoresDesviaciones_C1)
                print(vectoresDesviaciones_C2)

            numExperimentos = shape ( aplanado_SujetoA_SesionesB_C1[0][0] )[1]
            aceptados = calcularAceptados(  [ aplanado_SujetoA_SesionesB_C1, aplanado_SujetoA_SesionesB_C2 ],
                                            [ vectoresMedias_C1, vectoresMedias_C2 ],
                                            [ vectoresDesviaciones_C1, vectoresDesviaciones_C2 ],
                                            fronteraCorrelacion, fronteraProbabilidad, fronteraKurtosis, fronteraSkew,
                                            numExperimentos, verbose )

            rechazados = numExperimentos-aceptados
            print ( "Sujeto A: %i, Sujeto B: %i. Aceptado: %i (%.2f%%)\tRechazados: %i (%.2f%%)" % ( sujetoA, sujetoB, aceptados, aceptados * 100 / numExperimentos, rechazados, rechazados * 100 / numExperimentos )   )

#main()