# Identificacion de un sujeto mediante correlacion
# Programa de honores UDLAP


from warnings               import  filterwarnings
from numpy                  import  std, shape, divide
from scipy.stats            import  iqr
from CorrelacionCombinado   import  printC, cargarGDFMultiplesSesiones, \
                                    extraerDatosCrudosGDFMultiplesSesiones,\
                                    extraerCodificacionesCorteMultiplesSesiones, \
                                    extraerPuntosCorteMultiplesSesiones, \
                                    extraerExperimentosMultiplesSesiones, \
                                    filtrarPorColeccionesMultibanda
from Correlacion            import extraerCaracteristicas
from dimFractal             import katz
from MuestraDatos           import MAD
from matplotlib.pyplot      import plot, subplots, figure, show
from mpl_toolkits.mplot3d   import Axes3D



def normalizarSenal ( senal, canalA, canalB ):

    return divide( senal[:,canalA,:,:], senal[:,canalB,:,:] )

"""
Dividir las senales C1 entre C2 punto a punto
Input:  senalesC1 y senalesC2 con las senales filtradas
Output: senal normalizada
"""
def normalizarSenalesColeccionesBandas( senales, canalA, canalB ):

    return [ normalizarSenal( senal, canalA, canalB ) for senal in senales ]

def extraerCaracteristicasNormalizadas( senales, caracteristicas ):

    return [ [ extraerCaracteristicas( senal, caracteristica, list( range( 1, shape ( senal )[0] + 1 ) ) )
               for caracteristica in caracteristicas ] for senal in senales  ]

def mostrarVectores( valoresCaracteristicasA, valoresCaracteristicasB, caracteristicas, banda, angle=90, titulo=''):

    numCaracteristicas = len( valoresCaracteristicasA )

    fig = figure()
    fig.suptitle( titulo , fontsize=16)

    for numCaracteristica in range( numCaracteristicas ):

        ax = fig.add_subplot( 100 + 10 * numCaracteristicas + 1 + numCaracteristica , projection='3d' )
        ax.view_init(30, angle)
        ax.title.set_text( "\n\n" + caracteristicas[numCaracteristica].__name__)
        ax.set_xlabel(str(banda[0]))
        ax.set_ylabel(str(banda[1]))
        ax.set_zlabel(str(banda[2]))

        x = valoresCaracteristicasA[numCaracteristica][0,:]
        y = valoresCaracteristicasA[numCaracteristica][1,:]
        z = valoresCaracteristicasA[numCaracteristica][2,:]

        ax.scatter(x, y, z, c = 'r', marker='.')

        x = valoresCaracteristicasB[numCaracteristica][0,:]
        y = valoresCaracteristicasB[numCaracteristica][1,:]
        z = valoresCaracteristicasB[numCaracteristica][2,:]

        ax.scatter(x, y, z, c = 'b', marker='.')


    return ax



"""
Metodo principal
Input:  None
Output: None
"""
def main():

    sujetoA                     = 2                     # Primer sujeto a comparar [Sujeto de acceso]
    sujetoB                     = 9                     # Segundo sujeto a comparar [Sujeto de entrenamiento]
    sesionesEntrenar            = [2,3]                 # Sesiones empleadas en el entrenamiento del sistema
    sesionesClasificar          = [1]                   # Sesiones empleadas en la autenticacion
    tiempoConsiderado           = 2                     # Segundos considerados por experimento
    verbose                     = True                  # Mostrar procedimientos
    Fs                          = 250                   # Frecuencia de muestreo
    canalNormalizacionA         = 1                     # Numerador de la division de canales para normalizacion
    canalNormalizacionB         = 2                     # Denominador de la division de canales para normalizacion
    funcionesCaracteristicas    = [std, MAD, iqr]       # Funciones para la extraccion de caracteristicas

    coleccionesBandasC1         = [ [ (36, 43), (4, 8), (23, 35)  ],#, (20, 23), (18, 22) ],
                                    [ (39, 40), (41, 47), (47, 48) ] ]

    coleccionesBandasC2         = [ [ (36, 43), (4, 8), (23, 35) ], #(20, 23), (18, 22) ],
                                    [ (3, 6), (39, 40), (41, 45) ] ]

    # Suprimir alertas de la libreria MNE
    filterwarnings("ignore", category=Warning)

    if verbose:
        print ( "Cargando archivos GDF..." )

    # Cargar los datos de los dos sujetos en las sesiones de entrenamiento
    gdfs_SujetoA_SesionesA = cargarGDFMultiplesSesiones( sujetoA, sesionesEntrenar, "BCICIV_2b" )
    gdfs_SujetoB_SesionesA = cargarGDFMultiplesSesiones( sujetoB, sesionesEntrenar, "BCICIV_2b" )

    # Cargar los datos de los dos sujetos en las sesiones de clasificacion
    gdfs_SujetoA_SesionesB = cargarGDFMultiplesSesiones( sujetoA, sesionesClasificar, "BCICIV_2b" )
    gdfs_SujetoB_SesionesB = cargarGDFMultiplesSesiones( sujetoB, sesionesClasificar, "BCICIV_2b" )


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
        print ( "Sesiones B.\tSujeto A:\t", shapeDatosSujetoA_SesionesB,"\t"*5 +"Sujeto B:\t", shapeDatosSujetoB_SesionesB )


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
    puntosCorte_SujetoA_SesionesA = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoA_SesionesA )
    puntosCorte_SujetoB_SesionesA = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoB_SesionesA )
    puntosCorte_SujetoA_SesionesB = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoA_SesionesB )
    puntosCorte_SujetoB_SesionesB = extraerPuntosCorteMultiplesSesiones( gdfs_SujetoB_SesionesB )


    # Cortar senal
    # En este punto se pierden las listas de multi-sesion y se converge en
    # la forma (canales, muestras, exp)
    datosC1_SujetoA_SesionesA, datosC2_SujetoA_SesionesA = extraerExperimentosMultiplesSesiones( datosCrudosSujetoA_SesionesA,
                                                                                                 puntosCorte_SujetoA_SesionesA,
                                                                                                 codificacionCortes_SujetoA_SesionesA,
                                                                                                 tiempoConsiderado, False )

    datosC1_SujetoB_SesionesA, datosC2_SujetoB_SesionesA = extraerExperimentosMultiplesSesiones( datosCrudosSujetoB_SesionesA,
                                                                                                 puntosCorte_SujetoB_SesionesA,
                                                                                                 codificacionCortes_SujetoB_SesionesA,
                                                                                                 tiempoConsiderado, False )

    datosC1_SujetoA_SesionesB, datosC2_SujetoA_SesionesB = extraerExperimentosMultiplesSesiones( datosCrudosSujetoA_SesionesB,
                                                                                                 puntosCorte_SujetoA_SesionesB,
                                                                                                 codificacionCortes_SujetoA_SesionesB,
                                                                                                 tiempoConsiderado, False )

    datosC1_SujetoB_SesionesB, datosC2_SujetoB_SesionesB = extraerExperimentosMultiplesSesiones( datosCrudosSujetoB_SesionesB,
                                                                                                 puntosCorte_SujetoB_SesionesB,
                                                                                                 codificacionCortes_SujetoB_SesionesB,
                                                                                                 tiempoConsiderado, False )


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
        print("Datos filtrados C2\t Sesion A- A:", shapeFiltrado_SujetoA_SesionesA_C2, "\tB:", shapeFiltrado_SujetoB_SesionesA_C2 )
        print("Datos filtrados C2\t Sesion B- A:", shapeFiltrado_SujetoA_SesionesB_C2, "\tB:", shapeFiltrado_SujetoB_SesionesB_C2 )


    # Normalizar senal dividiendo entre canales
    senalNormalizada_SujetoA_SesionesA_C1 = normalizarSenalesColeccionesBandas( filtrado_SujetoA_SesionesA_C1, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoB_SesionesA_C1 = normalizarSenalesColeccionesBandas( filtrado_SujetoB_SesionesA_C1, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoA_SesionesB_C1 = normalizarSenalesColeccionesBandas( filtrado_SujetoA_SesionesB_C1, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoB_SesionesB_C1 = normalizarSenalesColeccionesBandas( filtrado_SujetoB_SesionesB_C1, canalNormalizacionA, canalNormalizacionB )

    senalNormalizada_SujetoA_SesionesA_C2 = normalizarSenalesColeccionesBandas( filtrado_SujetoA_SesionesA_C2, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoB_SesionesA_C2 = normalizarSenalesColeccionesBandas( filtrado_SujetoB_SesionesA_C2, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoA_SesionesB_C2 = normalizarSenalesColeccionesBandas( filtrado_SujetoA_SesionesB_C2, canalNormalizacionA, canalNormalizacionB )
    senalNormalizada_SujetoB_SesionesB_C2 = normalizarSenalesColeccionesBandas( filtrado_SujetoB_SesionesB_C2, canalNormalizacionA, canalNormalizacionB )

    # Mostrar normalizado
    if verbose:

        printC('BLUE', "\nNormalizacion de la senal")
        shapeSenalNormalizada_SujetoA_SesionesA_C1 = [ shape(datos) for datos in senalNormalizada_SujetoA_SesionesA_C1 ]
        shapeSenalNormalizada_SujetoB_SesionesA_C1 = [ shape(datos) for datos in senalNormalizada_SujetoB_SesionesA_C1 ]
        shapeSenalNormalizada_SujetoA_SesionesB_C1 = [ shape(datos) for datos in senalNormalizada_SujetoA_SesionesB_C1 ]
        shapeSenalNormalizada_SujetoB_SesionesB_C1 = [ shape(datos) for datos in senalNormalizada_SujetoB_SesionesB_C1 ]
        print("Senal normalizada C1\t Sesion A- A:", shapeSenalNormalizada_SujetoA_SesionesA_C1, "\tB:", shapeSenalNormalizada_SujetoB_SesionesA_C1 )
        print("Senal normalizada C1\t Sesion B- A:", shapeSenalNormalizada_SujetoA_SesionesB_C1, "\t\tB:", shapeSenalNormalizada_SujetoB_SesionesB_C1 )

        shapeSenalNormalizada_SujetoA_SesionesA_C2 = [ shape(datos) for datos in senalNormalizada_SujetoA_SesionesA_C2 ]
        shapeSenalNormalizada_SujetoB_SesionesA_C2 = [ shape(datos) for datos in senalNormalizada_SujetoB_SesionesA_C2 ]
        shapeSenalNormalizada_SujetoA_SesionesB_C2 = [ shape(datos) for datos in senalNormalizada_SujetoA_SesionesB_C2 ]
        shapeSenalNormalizada_SujetoB_SesionesB_C2 = [ shape(datos) for datos in senalNormalizada_SujetoB_SesionesB_C2 ]
        print("Senal normalizada C2\t Sesion A- A:", shapeSenalNormalizada_SujetoA_SesionesA_C2, "\tB:", shapeSenalNormalizada_SujetoB_SesionesA_C2 )
        print("Senal normalizada C2\t Sesion B- A:", shapeSenalNormalizada_SujetoA_SesionesB_C2, "\t\tB:", shapeSenalNormalizada_SujetoB_SesionesB_C2 )


    # Extraer caracteristicas
    caracteristicas_SujetoA_SesionesA_C1 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoA_SesionesA_C1, funcionesCaracteristicas )
    caracteristicas_SujetoB_SesionesA_C1 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoB_SesionesA_C1, funcionesCaracteristicas )
    caracteristicas_SujetoA_SesionesB_C1 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoA_SesionesB_C1, funcionesCaracteristicas )
    caracteristicas_SujetoB_SesionesB_C1 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoB_SesionesB_C1, funcionesCaracteristicas )

    caracteristicas_SujetoA_SesionesA_C2 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoA_SesionesA_C2, funcionesCaracteristicas )
    caracteristicas_SujetoB_SesionesA_C2 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoB_SesionesA_C2, funcionesCaracteristicas )
    caracteristicas_SujetoA_SesionesB_C2 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoA_SesionesB_C2, funcionesCaracteristicas )
    caracteristicas_SujetoB_SesionesB_C2 = extraerCaracteristicasNormalizadas( senalNormalizada_SujetoB_SesionesB_C2, funcionesCaracteristicas )

    if verbose:

        printC('BLUE', "\nExtraccion de caracteristicas: " + str( [ funcion.__name__ for funcion in funcionesCaracteristicas] ) )
        shapeCaracteristicas_SujetoA_SesionesA_C1 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoA_SesionesA_C1 ]
        shapeCaracteristicas_SujetoB_SesionesA_C1 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoB_SesionesA_C1 ]
        shapeCaracteristicas_SujetoA_SesionesB_C1 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoA_SesionesB_C1 ]
        shapeCaracteristicas_SujetoB_SesionesB_C1 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoB_SesionesB_C1 ]
        print("Caracteristicas C1\t Sesion A- A:", shapeCaracteristicas_SujetoA_SesionesA_C1, "\tB:", shapeCaracteristicas_SujetoB_SesionesA_C1 )
        print("Caracteristicas C1\t Sesion B- A:", shapeCaracteristicas_SujetoA_SesionesB_C1, "\t\tB:", shapeCaracteristicas_SujetoB_SesionesB_C1 )

        shapeCaracteristicas_SujetoA_SesionesA_C2 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoA_SesionesA_C2 ]
        shapeCaracteristicas_SujetoB_SesionesA_C2 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoB_SesionesA_C2 ]
        shapeCaracteristicas_SujetoA_SesionesB_C2 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoA_SesionesB_C2 ]
        shapeCaracteristicas_SujetoB_SesionesB_C2 = [ [shape(caract) for caract in banda ] for banda in caracteristicas_SujetoB_SesionesB_C2 ]
        print("Caracteristicas C2\t Sesion A- A:", shapeCaracteristicas_SujetoA_SesionesA_C2, "\tB:", shapeCaracteristicas_SujetoB_SesionesA_C2 )
        print("Caracteristicas C2\t Sesion B- A:", shapeCaracteristicas_SujetoA_SesionesB_C2, "\t\tB:", shapeCaracteristicas_SujetoB_SesionesB_C2 )


        # Visualizacion

        # Sujetos en sesion A, C1. Coleccion de bandas 1
        mostrarVectores(    caracteristicas_SujetoA_SesionesA_C1[0],
                            caracteristicas_SujetoB_SesionesA_C1[0],
                            funcionesCaracteristicas,
                            coleccionesBandasC1[0],
                            angle=45,
                            titulo = 'Sesion A, C1. Coleccion de bandas 1')

        # Sujetos en sesion A, C1. Coleccion de bandas 2
        mostrarVectores(    caracteristicas_SujetoA_SesionesA_C1[1],
                            caracteristicas_SujetoB_SesionesA_C1[1],
                            funcionesCaracteristicas,
                            coleccionesBandasC1[0],
                            angle=45,
                            titulo = 'Sesion A, C1. Coleccion de bandas 2')

        # Sujetos en sesion B, C1. Coleccion de bandas 1
        mostrarVectores(caracteristicas_SujetoA_SesionesB_C1[0],
                        caracteristicas_SujetoB_SesionesB_C1[0],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo = 'Sesion B, C1. Coleccion de bandas 1')

        # Sujetos en sesion B, C1. Coleccion de bandas 2
        mostrarVectores(caracteristicas_SujetoA_SesionesB_C1[1],
                        caracteristicas_SujetoB_SesionesB_C1[1],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo = 'Sesion B, C1. Coleccion de bandas 2')

        # --------------------- C2 ---------------------
        # Sujetos en sesion A, C1. Coleccion de bandas 1
        mostrarVectores(caracteristicas_SujetoA_SesionesA_C2[0],
                        caracteristicas_SujetoB_SesionesA_C2[0],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo = 'Sesion A, C2. Coleccion de bandas 1')

        # Sujetos en sesion A, C1. Coleccion de bandas 2
        mostrarVectores(caracteristicas_SujetoA_SesionesA_C2[1],
                        caracteristicas_SujetoB_SesionesA_C2[1],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo='Sesion A, C2. Coleccion de bandas 2')

        # Sujetos en sesion B, C1. Coleccion de bandas 1
        mostrarVectores(caracteristicas_SujetoA_SesionesB_C2[0],
                        caracteristicas_SujetoB_SesionesB_C2[0],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo='Sesion B, C2. Coleccion de bandas 1')

        # Sujetos en sesion B, C1. Coleccion de bandas 2
        mostrarVectores(caracteristicas_SujetoA_SesionesB_C2[1],
                        caracteristicas_SujetoB_SesionesB_C2[1],
                        funcionesCaracteristicas,
                        coleccionesBandasC1[0],
                        angle=45,
                        titulo='Sesion B, C2. Coleccion de bandas 2')

        show()

# Llamada metodo principal
main()
