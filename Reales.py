from scipy.io               import  loadmat
from numpy                  import  shape, concatenate, std, median,array, delete, mean, take_along_axis
from CorrelacionCombinado   import  filtrarPorColeccionesMultibanda, printC,\
                                    extraerMultiplesCaracteristicasColeccionesMultibanda,\
                                    ramoverOutliersMultiplesCaracteristicasColeccionesMultibanda,\
                                    aplanarMultiplesCaracteristicasColeccionesMultibanda, \
                                    obtenerFronteras, calcularAceptados,\
                                    normalizarMultiplesCaracteristicasColeccionesMultibanda
from scipy.stats            import  iqr, pearsonr
from MuestraDatos           import  MAD, localizarIndicesConservar
from matplotlib.pyplot      import  plot, show
from math                   import  inf
from numpy.random           import  rand

"""
El metodo permite cargar los datos en formato .mat de dos sujetos en diferentes
sesiones en la misma clase 
Input:  sujetoA :           (int) con el numero del sujeto que accede
        sujetoB :           (int) con el numero del sujeto con el que se entrena
        sesionClasificar :  (str) con la sesion que usara el sujeto A para acceder
        sesionesEntrenar :  [lista de str] con las sesiones para el entrenamiento
        claseUtilizada :    nombre de la clase del movimiento ej. class1 o class2
Output: tupla con numpy arrays 
"""
def cargar( sujetoA, sujetoB, sesionClasificar, sesionesEntrenar, claseUtilizada ):

    # Carga del sujeto A
    sujetoA_nombreArchivo = "./Data/A" + str(sujetoA) + "_S1" + str( sesionClasificar )
    sujetoA_datos = loadmat(sujetoA_nombreArchivo + ".mat")[claseUtilizada]

    # Inicializar valores para el sujeto B
    sujetoB_datos = None
    sujetoB_estaVacio = True

    # Por cada sesion indicada por el usuario
    for sesion in sesionesEntrenar:

        # Cargarla y ver si ya el np.array sujetoB_datos esta inicializado
        # Si es asi, concatenar. De lo contrario, inicializarlo
        sujetoB_nombreArchivo = "./Data/A" + str(sujetoB) + "_S1" + str( sesion )

        if sujetoB_estaVacio:
            sujetoB_estaVacio = False
            sujetoB_datos = loadmat(sujetoB_nombreArchivo + ".mat")[claseUtilizada]
        else:
            nuevosDatos = loadmat(sujetoB_nombreArchivo + ".mat")[claseUtilizada]
            datosConcatenados = concatenate( ( sujetoB_datos , nuevosDatos  ), -1)
            sujetoB_datos = datosConcatenados

    # Devolver tupla. El primer elemento contiene los datos de un sujeto,
    # el segundo contiene datos de los sujetos restantes (e.g. 2-9)
    return ( sujetoA_datos, sujetoB_datos )


"""
El metodo permite remover outliers mediante el metodo MAD
Input:  datos:          estructura en la forma [[(canal, exp), (canal, exp)], [(canal, exp), (canal, exp)]]
        desplazamiento: (float) con la cantidad de desplazamiento maximo permitido de la media
                        para no ser considerado outlier
Output: Estructura con la misma forma de entrada pero reducida en su numero
        de experimentos
"""
def filtrarMAD( datos, desplazamiento ):

    # Inicializar set de indices a eliminar
    puntosEliminar = set([])

    # Extraer el numero de experimentos y canales.
    # Se asume que se tiene el mismo numero en todos
    n_canales, n_exp = shape(datos[0][0][0])

    # Iterar sobre cada movimiento, colecciones de bandas
    # de frecuencia y caracteristica
    for movimiento in datos:
        for banda in movimiento:
            for caracteristica in banda:

                # Ahora que cada caracteristica es un array (canales, exp)

                # Calcular en cada canal los umbrales de filtrado mediana +- MAD * desplazamiento
                umbrales = [(median(array(caracteristica)[canal,:]) - (desplazamiento * MAD(array(caracteristica)[canal,:])),
                            median(array(caracteristica)[canal,:]) + (desplazamiento * MAD(array(caracteristica)[canal,:])))
                            for canal in range(n_canales)]

                # Con los umbrales ya calculados:
                # Iterar sobre todos los experimentos
                for exp in range(n_exp):

                    # Ver si en alguno de los canales se sale de los umbrales
                    for canal in range(n_canales):
                        if caracteristica[canal][exp] < umbrales[canal][0] or caracteristica[canal][exp] > umbrales[canal][1]:

                            # Si se sale en al menos un canal del umbral
                            # agregar el numero de experimento al set de exp por eliminar
                            puntosEliminar.add(exp)
                            break

    # Realizar la eliminacion de los mismos experimentos en todas las colecciones
    return [[[ delete(caracteristica,list(puntosEliminar),axis=1)   for caracteristica in banda ]
                                                                    for banda in movimiento]
                                                                    for movimiento in datos]


def main():

    for sujetoA in range(1,5):
        for sujetoB in range(1,5):

            # Parametros del experimento

            Fs = 125                                            # Frecuencia de muestreo
            sesionesEntrenamiento = ['a','b']                   # Sesiones utilizadas para entrenar
            sesionAutenticacion = 'a'                           # Sesion de autenticacion
            sujetoA = sujetoA                                   # Sujeto que accede
            sujetoB = sujetoB                                   # Sujeto cuyos datos son usados para entrenar
            coleccionesBandasC1 = [[(36, 43), (4,8), (23, 35),  # Bandas de filtrado generales
                                    (20, 23), (18, 22)]]
            coleccionesBandasC2 = [[(36, 43), (4,8), (23, 35),
                                    (20, 23), (18, 22)]]
            funcionesCaracteristicas = [std, MAD, iqr]          # Funciones para extraccion
            verbose = False                                     # Desplegar o no el procedimiento
            canales = [1,2,3]                                   # Canales de recopilacion
            desplazamiento = 3.5                                # Desplazamiento de MAD
            experimentosEntrenar = 2                            # Experimentos para entrenar
            anularKurtosisAsimetria = True                      # Anular las fronteras de kurtosis y asimetria
            correlacionEstatica = None                          # Correlacion de frontera.

            # Cargar los datos de los sujetos en ambas clases
            datosC1_SujetoA_SesionA, datosC1_SujetoB_SesionesB = cargar( sujetoA, sujetoB, sesionAutenticacion, sesionesEntrenamiento, "class1" )
            datosC2_SujetoA_SesionA, datosC2_SujetoB_SesionesB = cargar( sujetoA, sujetoB, sesionAutenticacion, sesionesEntrenamiento, "class2" )

            # Filtrar datos de ambos sujetos en ambas clases
            filtrado_SujetoA_SesionA_C1 =   filtrarPorColeccionesMultibanda( datosC1_SujetoA_SesionA, coleccionesBandasC1, Fs)
            filtrado_SujetoB_SesionesB_C1 = filtrarPorColeccionesMultibanda(datosC1_SujetoB_SesionesB, coleccionesBandasC1, Fs)
            filtrado_SujetoA_SesionA_C2 =   filtrarPorColeccionesMultibanda( datosC2_SujetoA_SesionA, coleccionesBandasC2, Fs)
            filtrado_SujetoB_SesionesB_C2 = filtrarPorColeccionesMultibanda( datosC2_SujetoB_SesionesB, coleccionesBandasC2, Fs)

            # Mostrar datos filtrados a peticion del usuario
            if verbose:
                printC('BLUE', "Filtrado de datos")
                shapeFiltrado_SujetoA_SesionA_C1    = [ shape(datos) for datos in filtrado_SujetoA_SesionA_C1 ]
                shapeFiltrado_SujetoA_SesionA_C2    = [ shape(datos) for datos in filtrado_SujetoA_SesionA_C2 ]
                shapeFiltrado_SujetoB_SesionesB_C1  = [ shape(datos) for datos in filtrado_SujetoB_SesionesB_C1 ]
                shapeFiltrado_SujetoB_SesionesB_C2  = [ shape(datos) for datos in filtrado_SujetoB_SesionesB_C2 ]
                print("Datos filtrados C1\t Sesion A- A:", shapeFiltrado_SujetoA_SesionA_C1, "\t Sesiones B- B:", shapeFiltrado_SujetoB_SesionesB_C1)
                print("Datos filtrados C2\t Sesion A- A:", shapeFiltrado_SujetoA_SesionA_C2, "\t Sesiones B- B:", shapeFiltrado_SujetoB_SesionesB_C2)

            # Realizar extraccion de caracteristicas
            caracteristicas_SujetoA_SesionA_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionA_C1, funcionesCaracteristicas, canales )
            caracteristicas_SujetoB_SesionesB_C1 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesB_C1, funcionesCaracteristicas, canales )
            caracteristicas_SujetoA_SesionA_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoA_SesionA_C2, funcionesCaracteristicas, canales )
            caracteristicas_SujetoB_SesionesB_C2 = extraerMultiplesCaracteristicasColeccionesMultibanda( filtrado_SujetoB_SesionesB_C2, funcionesCaracteristicas, canales )

            # Mostrar forma de los datos tras la extraccion a peticion del usuario
            if verbose:
                printC('BLUE', "\nExtraccion de caracteristicas: " +
                       str([funcion.__name__ for funcion in funcionesCaracteristicas]))
                shapeCaracteristicas_SujetoA_SesionA_C1= [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in caracteristicas_SujetoA_SesionA_C1]
                shapeCaracteristicas_SujetoB_SesionesB_C1 = [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in caracteristicas_SujetoB_SesionesB_C1]
                shapeCaracteristicas_SujetoA_SesionA_C2= [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in caracteristicas_SujetoA_SesionA_C2]
                shapeCaracteristicas_SujetoB_SesionesB_C2= [[shape(caracteristica)  for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in caracteristicas_SujetoB_SesionesB_C2]


                print("Caracteristicas C1. Sesion A-\tA:", shapeCaracteristicas_SujetoA_SesionA_C1, "\tSesiones B:",
                      shapeCaracteristicas_SujetoB_SesionesB_C1)
                print("Caracteristicas C2. Sesion B-\tA:", shapeCaracteristicas_SujetoA_SesionA_C2, "\tSesiones B:",
                      shapeCaracteristicas_SujetoB_SesionesB_C2)

            # Aplanar datos
            aplanado_SujetoA_SesionA_C1 =   aplanarMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionA_C1)
            aplanado_SujetoB_SesionesB_C1 = aplanarMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesB_C1)
            aplanado_SujetoA_SesionA_C2 =   aplanarMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoA_SesionA_C2)
            aplanado_SujetoB_SesionesB_C2 = aplanarMultiplesCaracteristicasColeccionesMultibanda(caracteristicas_SujetoB_SesionesB_C2)

            # Mostrar forma de los datos tras el aplanado a peticion del usuario
            if verbose:
                printC('BLUE', "\nAplanado de datos")
                shapeAplanado_SujetoA_SesionA_C1 = [[shape(caracteristica)  for caracteristica  in coleccionBandas]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionA_C1]
                shapeAplanado_SujetoB_SesionesB_C1= [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C1]
                shapeAplanado_SujetoA_SesionA_C2 = [[shape(caracteristica)  for caracteristica  in coleccionBandas]
                                                                            for coleccionBandas in aplanado_SujetoA_SesionA_C2]
                shapeAplanado_SujetoB_SesionesB_C2= [[shape(caracteristica) for caracteristica  in coleccionBandas]
                                                                            for coleccionBandas in aplanado_SujetoB_SesionesB_C2]

                print("Aplanado C1. Sesion A-\tA:", shapeAplanado_SujetoA_SesionA_C1, "\tSesiones B:",
                      shapeAplanado_SujetoB_SesionesB_C1)
                print("Aplanado C2. Sesion B-\tA:", shapeAplanado_SujetoA_SesionA_C2, "\tSesiones B:",
                      shapeAplanado_SujetoB_SesionesB_C2)

            # Remocion de outliers
            sinOutliers_SujetoA_SesionA_C1, \
            sinOutliers_SujetoA_SesionA_C2 = filtrarMAD( [aplanado_SujetoA_SesionA_C1,
                                                          aplanado_SujetoA_SesionA_C2], desplazamiento )

            sinOutliers_SujetoB_SesionesB_C1, \
            sinOutliers_SujetoB_SesionesB_C2= filtrarMAD( [aplanado_SujetoB_SesionesB_C1,
                                                            aplanado_SujetoB_SesionesB_C2], desplazamiento)

            # Mostrar forma de los datos tras la remocion a peticion del usuario
            if verbose:
                printC('BLUE', "\nRemovedor de outliers. MAD " + str(desplazamiento) )
                shapeSinOutliers_SujetoA_SesionA_C1    = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in sinOutliers_SujetoA_SesionA_C1]
                shapeSinOutliers_SujetoB_SesionesB_C1  = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in sinOutliers_SujetoB_SesionesB_C1]
                shapeSinOutliers_SujetoA_SesionA_C2    = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in sinOutliers_SujetoA_SesionA_C2]
                shapeSinOutliers_SujetoB_SesionesB_C2  = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                                    for coleccionBandas in sinOutliers_SujetoB_SesionesB_C2]

                print("Sin outliers C1. Sesion A-\tA:", shapeSinOutliers_SujetoA_SesionA_C1, "\tSesiones B:", shapeSinOutliers_SujetoB_SesionesB_C1)
                print("Sin outliers C2. Sesion B-\tA:", shapeSinOutliers_SujetoA_SesionA_C2, "\tSesiones B:", shapeSinOutliers_SujetoB_SesionesB_C2)


            # Calcular vectores de entrenamiento
            vectoresMedias_C1, vectoresDesviaciones_C1, sinOutliers_SujetoB_SesionesB_C1 = calcularParametrosEntrenamiento( sinOutliers_SujetoB_SesionesB_C1, experimentosEntrenar )
            vectoresMedias_C2, vectoresDesviaciones_C2, sinOutliers_SujetoB_SesionesB_C2 = calcularParametrosEntrenamiento( sinOutliers_SujetoB_SesionesB_C2, experimentosEntrenar )

            # Mostrar forma de los vectores de entrenamiento a peticion del usuario
            if verbose:

                printC('BLUE', "\nEntrenamiento. SujetoA SesionA -> Accede. SujetoB SesionesB -> Entrena")

                shapeVectoresMedias_C1 = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresMedias_C1]
                shapeDesviaciones_C1   = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresDesviaciones_C1]
                shapeVectoresMedias_C2 = [[shape(caracteristica)    for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresMedias_C2]
                shapeDesviaciones_C2 = [[shape(caracteristica)      for caracteristica  in coleccionBandas]
                                                                    for coleccionBandas in vectoresDesviaciones_C2]

                print("Vectores medios C1\t\t- B:", shapeVectoresMedias_C1)
                print("Vectores desviacion C1\t- B:", shapeDesviaciones_C1)
                print("Vectores medios C2\t\t- B:", shapeVectoresMedias_C2)
                print("Vectores desviacion C2\t- B:", shapeDesviaciones_C2)

            # Identificador de fronteras
            fronteraCorrelacion, fronteraProbabilidad, \
            fronteraKurtosis, fronteraSkew = obtenerFronteras(  [ sinOutliers_SujetoB_SesionesB_C1, sinOutliers_SujetoB_SesionesB_C2 ],
                                                                [ vectoresMedias_C1, vectoresMedias_C2 ],
                                                                [ vectoresDesviaciones_C1, vectoresDesviaciones_C2 ], correlacionEstatica,
                                                                anularKurtosisAsimetria)

            # Mostrar un experimento a peticion del usuario
            if verbose:
                plot(array(sinOutliers_SujetoA_SesionA_C1[0][0])[:,1],'k')
                plot(vectoresMedias_C1[0][0],'r')
                show()

            # Calcular numero de aceptaciones y rechazos
            numExperimentos = shape ( sinOutliers_SujetoA_SesionA_C1[0][0] )[1]
            aceptados = calcularAceptados(  [ sinOutliers_SujetoA_SesionA_C1, sinOutliers_SujetoA_SesionA_C2 ],
                                            [ vectoresMedias_C1, vectoresMedias_C2 ],
                                            [ vectoresDesviaciones_C1, vectoresDesviaciones_C2 ],
                                            fronteraCorrelacion, fronteraProbabilidad, fronteraKurtosis, fronteraSkew,
                                            numExperimentos, verbose )

            # Desplegar resultados
            rechazados = numExperimentos-aceptados
            print ( "Sujeto A: %s, Sujeto B: %s. Aceptado: %i (%.2f%%)\tRechazados: %i (%.2f%%)" % ( sujetoA, sujetoB, aceptados, aceptados * 100 / numExperimentos, rechazados, rechazados * 100 / numExperimentos )   )


def calcularParametrosEntrenamiento( datos, expEntrenar ):

    medias = []
    desviaciones = []
    valores = []

    for coleccionBandas in datos:

        valoresCaracteristica = []
        mediasCaracteristica = []
        desviacionesCaracteristica = []

        for caracteristica in coleccionBandas:

            coleccionDatosMezclados = shuffle_along_axis( array(caracteristica), 1 )
            coleccionDatos = coleccionDatosMezclados[:,:expEntrenar]
            valoresCaracteristica.append(coleccionDatosMezclados[:,expEntrenar:])
            mediasCaracteristica.append( mean( coleccionDatos, axis = 1 ) )
            desviacionesCaracteristica.append( std( coleccionDatos, axis = 1 ) )

        valores.append(valoresCaracteristica)
        medias.append(mediasCaracteristica)
        desviaciones.append(desviacionesCaracteristica)

    return (medias, desviaciones, valores)

def shuffle_along_axis(a, axis):
    idx = rand(*a.shape).argsort(axis=axis)
    return take_along_axis(a,idx,axis=axis)

# Llamada al metodo principal
main()