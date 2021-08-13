# Identificacion de un sujeto mediante correlacion
# Programa de honores UDLAP

from entropy import *
from math                           import  inf
from dit.other                      import  renyi_entropy
from dit                            import  Distribution
from warnings                       import  filterwarnings
from mne.io                         import  read_raw_gdf
from mne                            import  events_from_annotations
from Correlacion                    import  filtrarMultibanda, extraerCaracteristicasMultibanda, \
                                            normalizarNumeroExperimentos, calcularAceptados
from numpy                          import  mean, std, shape, array, median, var, asarray, append, ndarray, \
                                            insert, vstack, reshape, transpose, fft
from scipy.io                       import  loadmat
from scipy.signal                   import  butter, lfilter
from scipy.stats                    import  norm, entropy, iqr
from scipy.stats.stats              import  pearsonr, spearmanr, kendalltau, kurtosis, skew
from dimFractal                     import  katz, hfd
from MuestraDatos                   import  ajustarNumeroExperimentos, MAD
from ClasificadorUnoTodosMultiBanda import  removerOutliersSujeto, \
                                            aplanarDimensionBandasFrecuenciaIndividual, \
                                            distribuirDatosEntrenamientoYClasificacion

#def spectral_centroid(x, samplerate=250):
#    magnitudes = abs(fft.rfft(x))
#    length = len(x)
#    freqs = abs(fft.fftfreq(length, 1.0/samplerate)[:length//2+1])
#    magnitudes = magnitudes[:length//2+1]
#    return sum(magnitudes*freqs) / sum(magnitudes)

#def renyi( datos, alpha=3 ):
#    renyi_entropy( Distribution( datos ), alpha )



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

    # Calcular muestras a desplazar a la izquierda
    # El documento indica que la duracion de un experimento es 4.5 seg
    desplazamientoTiempo = int ( ( duracionTotalExperimento- tiempo ) / 2 ) * Fs

    # Indicador para rechazar proximo experimento
    rechazar    = False

    # Almacenar muestras recortadas
    muestras_C1 = []
    muestras_C2 = []

    # Recorrer puntos de corte
    for puntoCorte in puntosCorte:

        # Sustraer datos del punto de corte
        muestra, _, codigo = puntoCorte

        # En caso de encontrar un codigo de flecha a la izquierda
        if codificacionCorte[ codigo ] == '769':
            if not rechazar:
                if tomarDesdeCruz:
                    muestra -= duracionCruz * Fs
                muestras_C1.append( senal[ : , muestra + desplazamientoTiempo : muestra + desplazamientoTiempo + int(tiempo * Fs) ] )
            rechazar = False

        # En caso de encontrar un codigo de flecha a la izquierda
        if codificacionCorte[ codigo ] == '770':
            if not rechazar:
                if tomarDesdeCruz:
                    muestra -= duracionCruz * Fs
                muestras_C2.append( senal[ : , muestra + desplazamientoTiempo : muestra + desplazamientoTiempo + int(tiempo * Fs) ] )
            rechazar = False

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
Metodo principal
Input:  None
Output: None
"""
def main():

    for sujetoA in range(1,10):
        for sujetoB in range(1,10):

            if ( sujetoB == 9 ):
                ubicacionDatos          = "BCICIV_2b"   # Nombre del directorio donde se localizan los datos de los sujetos
                #sujetoA                 = 1            # Primer sujeto a comparar
                #sujetoB                 = 9            # Segundo sujeto a comparar
                sesionA                 = 2             # Numero de la primera sesion (entrenamiento)
                sesionB                 = 3             # Numero de la segunda sesion (entrenamiento)
                canales                 = [1,2,3]       # Canales a considerar
                desplazamiento          = 3.5           # Veces (MAD) distanciamiento de la mediana
                Fs                      = 250           # Frecuencia de muestreo
                funcionCaracteristica   = iqr           # Funcion para la extraccion de caracteristicas
                tiempoConsiderado       = 2             # Segundos considerados por experimento
                tomarDesdeCruz          = False         # Si se toma el tiempo desde la cruz de ajuste o desde la flecha
                verbose                 = True          # Si se muestra el procedimiento

                frecuenciasFiltro_C1    = [ (39, 40), (41, 47), (47, 48) ]
                frecuenciasFiltro_C2    = [ (3, 6), (39, 40), (41, 45) ]

                # Suprimir alertas de la libreria MNE
                filterwarnings("ignore", category=Warning)

                if verbose:
                    print ( "Cargando archivos GDF..." )

                # Cargar los datos de los dos sujetos en la primera sesion
                gdf_SujetoA_SesionA = cargarGDFcrudo( sujetoA , sesionA , ubicacionDatos )
                gdf_SujetoB_SesionA = cargarGDFcrudo( sujetoB , sesionA , ubicacionDatos )

                # Cargar los datos de los dos sujetos en la segunda sesion
                gdf_SujetoA_SesionB = cargarGDFcrudo( sujetoA , sesionB , ubicacionDatos )
                gdf_SujetoB_SesionB = cargarGDFcrudo( sujetoB , sesionB , ubicacionDatos )

                if verbose:
                    print( "Carga completa" )

                # Extraer datos de los GDF
                # Extraer primeros 3 canales (los demas son de potenciales visuales evocados P300)
                datosCrudosSujetoA_SesionA = gdf_SujetoA_SesionA._data [:3,:]
                datosCrudosSujetoB_SesionA = gdf_SujetoB_SesionA._data [:3,:]
                datosCrudosSujetoA_SesionB = gdf_SujetoA_SesionB._data [:3,:]
                datosCrudosSujetoB_SesionB = gdf_SujetoB_SesionB._data [:3,:]

                if verbose:
                    printC( 'BLUE', "\nDatos cargados de los primeros" )
                    print ( "Sesion A.\tSujeto A:\t", shape( datosCrudosSujetoA_SesionA ),"\tSujeto B:\t", shape( datosCrudosSujetoB_SesionA ) )
                    print ( "Sesion B.\tSujeto A:\t", shape( datosCrudosSujetoA_SesionB ),"\tSujeto B:\t", shape( datosCrudosSujetoB_SesionB ) )


                    printC('BLUE', "\nExtraccion de los puntos de corte")

                # Extraer el esquema de codificacion de los puntos de corte
                # Se extrae del primer elemento, aunque cualquier elemento es util, ya que este es constante
                # en todos los sujetos
                codificacionCortes_SujetoA_SesionA = invertirDiccionario( events_from_annotations ( gdf_SujetoA_SesionA, verbose='ERROR' )[1] )
                codificacionCortes_SujetoB_SesionA = invertirDiccionario(  events_from_annotations( gdf_SujetoB_SesionA, verbose='ERROR')[1] )
                codificacionCortes_SujetoA_SesionB = invertirDiccionario(  events_from_annotations( gdf_SujetoA_SesionB, verbose='ERROR')[1] )
                codificacionCortes_SujetoB_SesionB = invertirDiccionario(  events_from_annotations( gdf_SujetoB_SesionB, verbose='ERROR')[1] )

                if verbose:
                    printC( 'BOLD', "Codificacion de puntos de corte:" )
                    print( codificacionCortes_SujetoA_SesionA )
                    print( codificacionCortes_SujetoB_SesionA )
                    print( codificacionCortes_SujetoA_SesionB )
                    print( codificacionCortes_SujetoB_SesionB )

                # Extraer listas con puntos de corte
                puntosCorte_SujetoA_SesionA = events_from_annotations( gdf_SujetoA_SesionA, verbose='ERROR' )[0]
                puntosCorte_SujetoB_SesionA = events_from_annotations( gdf_SujetoB_SesionA, verbose='ERROR' )[0]
                puntosCorte_SujetoA_SesionB = events_from_annotations( gdf_SujetoA_SesionB, verbose='ERROR' )[0]
                puntosCorte_SujetoB_SesionB = events_from_annotations( gdf_SujetoB_SesionB, verbose='ERROR' )[0]

                # Cortar senal
                datosC1_SujetoA_SesionA, datosC2_SujetoA_SesionA = extraerExperimentos( datosCrudosSujetoA_SesionA,
                                                                                        puntosCorte_SujetoA_SesionA,
                                                                                        codificacionCortes_SujetoA_SesionA,
                                                                                        tiempoConsiderado,
                                                                                        tomarDesdeCruz=tomarDesdeCruz )

                datosC1_SujetoB_SesionA, datosC2_SujetoB_SesionA = extraerExperimentos( datosCrudosSujetoB_SesionA,
                                                                                        puntosCorte_SujetoB_SesionA,
                                                                                        codificacionCortes_SujetoB_SesionA,
                                                                                        tiempoConsiderado,
                                                                                        tomarDesdeCruz=tomarDesdeCruz )

                datosC1_SujetoA_SesionB, datosC2_SujetoA_SesionB = extraerExperimentos( datosCrudosSujetoA_SesionB,
                                                                                        puntosCorte_SujetoA_SesionB,
                                                                                        codificacionCortes_SujetoA_SesionB,
                                                                                        tiempoConsiderado,
                                                                                        tomarDesdeCruz=tomarDesdeCruz)

                datosC1_SujetoB_SesionB, datosC2_SujetoB_SesionB = extraerExperimentos( datosCrudosSujetoB_SesionB,
                                                                                        puntosCorte_SujetoB_SesionB,
                                                                                        codificacionCortes_SujetoB_SesionB,
                                                                                        tiempoConsiderado,
                                                                                        tomarDesdeCruz=tomarDesdeCruz)

                # Mostrar cortes de senal
                if verbose:
                    printC('BLUE', "\nResultado de recorte de senal")
                    print ( "C1: Sesion A. Sujeto A", shape( datosC1_SujetoA_SesionA ), "\tSesion A. Sujeto B", shape( datosC1_SujetoB_SesionA ) )
                    print ( "C1: Sesion B. Sujeto A", shape( datosC1_SujetoA_SesionB ), "\tSesion B. Sujeto B", shape( datosC1_SujetoB_SesionB ) )
                    print ( "C2: Sesion A. Sujeto A", shape( datosC2_SujetoA_SesionA ), "\tSesion A. Sujeto B", shape( datosC2_SujetoB_SesionA ) )
                    print ( "C2: Sesion B. Sujeto A", shape( datosC2_SujetoA_SesionB ), "\tSesion B. Sujeto B", shape( datosC2_SujetoB_SesionB ) )


                # Filtrado

                if verbose:
                    printC('BLUE', "\nFiltrado de datos")
                filtrado_SujetoA_SesionA_C1 = filtrarMultibanda( datosC1_SujetoA_SesionA, frecuenciasFiltro_C1, Fs)
                filtrado_SujetoB_SesionA_C1 = filtrarMultibanda( datosC1_SujetoB_SesionA, frecuenciasFiltro_C1, Fs)
                filtrado_SujetoA_SesionB_C1 = filtrarMultibanda( datosC1_SujetoA_SesionB, frecuenciasFiltro_C1, Fs)
                filtrado_SujetoB_SesionB_C1 = filtrarMultibanda( datosC1_SujetoB_SesionB, frecuenciasFiltro_C1, Fs)

                if verbose:
                    print("Datos filtrados C1\t Sesion A- A:", shape( filtrado_SujetoA_SesionA_C1 ), "\tB:", shape( filtrado_SujetoB_SesionA_C1 ) )
                    print("Datos filtrados C1\t Sesion A- A:", shape( filtrado_SujetoA_SesionB_C1 ), "\tB:", shape( filtrado_SujetoB_SesionB_C1 ) )

                filtrado_SujetoA_SesionA_C2 = filtrarMultibanda( datosC2_SujetoA_SesionA, frecuenciasFiltro_C2, Fs)
                filtrado_SujetoB_SesionA_C2 = filtrarMultibanda( datosC2_SujetoB_SesionA, frecuenciasFiltro_C2, Fs)
                filtrado_SujetoA_SesionB_C2 = filtrarMultibanda( datosC2_SujetoA_SesionB, frecuenciasFiltro_C2, Fs)
                filtrado_SujetoB_SesionB_C2 = filtrarMultibanda( datosC2_SujetoB_SesionB, frecuenciasFiltro_C2, Fs)

                if verbose:
                    print("Datos filtrados C2. Sesion A-\tA:", shape( filtrado_SujetoA_SesionA_C2 ), "\tB:", shape( filtrado_SujetoB_SesionA_C2 ) )
                    print("Datos filtrados C2. Sesion A-\tA:", shape( filtrado_SujetoA_SesionB_C2 ), "\tB:", shape( filtrado_SujetoB_SesionB_C2 ) )

                # Extraccion de caracteristicas

                if verbose:
                    printC('BLUE', "\nExtraccion de caracteristicas")
                caracteristicas_SujetoA_SesionA_C1 = extraerCaracteristicasMultibanda( filtrado_SujetoA_SesionA_C1, funcionCaracteristica, canales )
                caracteristicas_SujetoB_SesionA_C1 = extraerCaracteristicasMultibanda( filtrado_SujetoB_SesionA_C1, funcionCaracteristica, canales )
                caracteristicas_SujetoA_SesionB_C1 = extraerCaracteristicasMultibanda( filtrado_SujetoA_SesionB_C1, funcionCaracteristica, canales )
                caracteristicas_SujetoB_SesionB_C1 = extraerCaracteristicasMultibanda( filtrado_SujetoB_SesionB_C1, funcionCaracteristica, canales )

                if verbose:
                    printC('BOLD', "Extraccion para C1. Funcion: " + funcionCaracteristica.__name__ )
                    print ("Caracteristicas C1. Sesion A-\tA:", shape( caracteristicas_SujetoA_SesionA_C1 ),
                                                        "\tB:", shape( caracteristicas_SujetoB_SesionA_C1 ) )
                    print ("Caracteristicas C1. Sesion B-\tA:", shape( caracteristicas_SujetoA_SesionB_C1 ),
                                                        "\tB:", shape( caracteristicas_SujetoB_SesionB_C1 ) )

                caracteristicas_SujetoA_SesionA_C2 = extraerCaracteristicasMultibanda( filtrado_SujetoA_SesionA_C2, funcionCaracteristica, canales )
                caracteristicas_SujetoB_SesionA_C2 = extraerCaracteristicasMultibanda( filtrado_SujetoB_SesionA_C2, funcionCaracteristica, canales )
                caracteristicas_SujetoA_SesionB_C2 = extraerCaracteristicasMultibanda( filtrado_SujetoA_SesionB_C2, funcionCaracteristica, canales )
                caracteristicas_SujetoB_SesionB_C2 = extraerCaracteristicasMultibanda( filtrado_SujetoB_SesionB_C2, funcionCaracteristica, canales )

                if verbose:
                    printC('BOLD', "Extraccion para C2. Funcion: " + funcionCaracteristica.__name__ )
                    print ("Caracteristicas C2. Sesion A-\tA:", shape( caracteristicas_SujetoA_SesionA_C2 ),
                                                        "\tB:", shape( caracteristicas_SujetoB_SesionA_C2 ) )
                    print ("Caracteristicas C2. Sesion B-\tA:", shape( caracteristicas_SujetoA_SesionB_C2 ),
                                                        "\tB:", shape( caracteristicas_SujetoB_SesionB_C2 ) )

                # Remocion de outliers
                if verbose:
                    printC('BLUE', "\nRemocion de outliers")
                    printC('BOLD', "Removedor de outliers para C1. MAD: " + str( desplazamiento ) )
                sinOutliers_SujetoA_SesionA_C1 = removerOutliersSujeto(caracteristicas_SujetoA_SesionA_C1, desplazamiento)
                sinOutliers_SujetoB_SesionA_C1 = removerOutliersSujeto(caracteristicas_SujetoB_SesionA_C1, desplazamiento)
                sinOutliers_SujetoA_SesionB_C1 = removerOutliersSujeto(caracteristicas_SujetoA_SesionB_C1, desplazamiento)
                sinOutliers_SujetoB_SesionB_C1 = removerOutliersSujeto(caracteristicas_SujetoB_SesionB_C1, desplazamiento)

                if verbose:
                    print ("Datos tras remocion C1. Sesion A-\tA:", shape( sinOutliers_SujetoA_SesionA_C1 ),
                                                            "\tB:", shape( sinOutliers_SujetoB_SesionA_C1 ) )
                    print ("Datos tras remocion C1. Sesion B-\tA:", shape( sinOutliers_SujetoA_SesionB_C1 ),
                                                            "\tB:", shape( sinOutliers_SujetoB_SesionB_C1 ) )

                    printC('BOLD', "Removedor de outliers para C2. MAD: " + str( desplazamiento ) )
                sinOutliers_SujetoA_SesionA_C2 = removerOutliersSujeto(caracteristicas_SujetoA_SesionA_C2, desplazamiento)
                sinOutliers_SujetoB_SesionA_C2 = removerOutliersSujeto(caracteristicas_SujetoB_SesionA_C2, desplazamiento)
                sinOutliers_SujetoA_SesionB_C2 = removerOutliersSujeto(caracteristicas_SujetoA_SesionB_C2, desplazamiento)
                sinOutliers_SujetoB_SesionB_C2 = removerOutliersSujeto(caracteristicas_SujetoB_SesionB_C2, desplazamiento)

                if verbose:
                    print ("Datos tras remocion C2. Sesion A-\tA:", shape( sinOutliers_SujetoA_SesionA_C2 ),
                                                            "\tB:", shape( sinOutliers_SujetoB_SesionA_C2 ) )
                    print ("Datos tras remocion C2. Sesion B-\tA:", shape( sinOutliers_SujetoA_SesionB_C2 ),
                                                            "\tB:", shape( sinOutliers_SujetoB_SesionB_C2 ) )

                # Aplanamiento de datos sin outliers
                    printC('BLUE', "\nAplanamiento de datos")
                    printC('BOLD', "Aplanamiento para datos de C1" )
                aplanado_SujetoA_SesionA_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoA_SesionA_C1 )
                aplanado_SujetoB_SesionA_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoB_SesionA_C1 )
                aplanado_SujetoA_SesionB_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoA_SesionB_C1 )
                aplanado_SujetoB_SesionB_C1 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoB_SesionB_C1 )

                if verbose:
                    print ("Datos aplanados C1. Sesion A-\tA:", shape( aplanado_SujetoA_SesionA_C1 ),
                                                        "\tB:", shape( aplanado_SujetoB_SesionA_C1 ) )
                    print ("Datos aplanados C1. Sesion B-\tA:", shape( aplanado_SujetoA_SesionB_C1 ),
                                                        "\tB:", shape( aplanado_SujetoB_SesionB_C1 ) )

                    printC('BOLD', "Aplanamiento para datos de C2" )
                aplanado_SujetoA_SesionA_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoA_SesionA_C2 )
                aplanado_SujetoB_SesionA_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoB_SesionA_C2 )
                aplanado_SujetoA_SesionB_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoA_SesionB_C2 )
                aplanado_SujetoB_SesionB_C2 = aplanarDimensionBandasFrecuenciaIndividual( sinOutliers_SujetoB_SesionB_C2 )

                if verbose:
                    print ("Datos aplanados C2. Sesion A-\tA:", shape( aplanado_SujetoA_SesionA_C2 ),
                                                        "\tB:", shape( aplanado_SujetoB_SesionA_C2 ) )
                    print ("Datos aplanados C2. Sesion B-\tA:", shape( aplanado_SujetoA_SesionB_C2 ),
                                                        "\tB:", shape( aplanado_SujetoB_SesionB_C2 ) )

                # Normalizacion del numero de experimentos
                    printC('BLUE', "\nNormalizacion del numero de datos")
                    print("Misma cantidad entre sujeto A y B y en clase C1 y C2 [Discriminacion con ambos]")
                normalizarNumeroExperimentos( [ aplanado_SujetoA_SesionA_C1, aplanado_SujetoB_SesionA_C1,
                                                aplanado_SujetoA_SesionA_C2, aplanado_SujetoB_SesionA_C2 ] )
                normalizarNumeroExperimentos( [ aplanado_SujetoA_SesionB_C2, aplanado_SujetoB_SesionB_C2,
                                                aplanado_SujetoA_SesionB_C1, aplanado_SujetoB_SesionB_C1] )

                if verbose:
                    print("Datos normalizados C1. Sesion A-\tA:", shape( aplanado_SujetoA_SesionA_C1 ),
                                                          "\tB:", shape( aplanado_SujetoB_SesionA_C1 ) )
                    print("Datos normalizados C1. Sesion B-\tA:", shape( aplanado_SujetoA_SesionB_C1 ),
                                                          "\tB:", shape( aplanado_SujetoB_SesionB_C1 ) )
                    print("Datos normalizados C2. Sesion A-\tA:", shape( aplanado_SujetoA_SesionA_C2 ),
                                                          "\tB:", shape( aplanado_SujetoB_SesionA_C2 ) )
                    print("Datos normalizados C2. Sesion B-\tA:", shape( aplanado_SujetoA_SesionB_C2 ),
                                                          "\tB:", shape( aplanado_SujetoB_SesionB_C2 ) )

                # Unidad de prueba. Garantizar que los datos no se cruzaron por error
                    printC('BLUE', "\nUnidad de prueba: No existe cruce de datos ")
                prueba_SujetoA_SesionA = aplanado_SujetoA_SesionA_C1 == aplanado_SujetoA_SesionA_C2
                prueba_SujetoB_SesionA = aplanado_SujetoB_SesionA_C1 == aplanado_SujetoB_SesionA_C2
                prueba_SujetoA_SesionB = aplanado_SujetoA_SesionB_C1 == aplanado_SujetoA_SesionB_C2
                prueba_SujetoB_SesionB = aplanado_SujetoB_SesionB_C1 == aplanado_SujetoB_SesionB_C2

                if verbose:
                    printC( 'RED' if prueba_SujetoA_SesionA else 'END' , "SujetoA. SesionA: " + ("[Fallida]" if prueba_SujetoA_SesionA else "[Aprobada]") )
                    printC( 'RED' if prueba_SujetoB_SesionA else 'END' , "SujetoB. SesionA: " + ("[Fallida]" if prueba_SujetoB_SesionA else "[Aprobada]") )
                    printC( 'RED' if prueba_SujetoA_SesionB else 'END' , "SujetoA. SesionB: " + ("[Fallida]" if prueba_SujetoA_SesionB else "[Aprobada]") )
                    printC( 'RED' if prueba_SujetoB_SesionB else 'END' , "SujetoB. SesionB: " + ("[Fallida]" if prueba_SujetoB_SesionB else "[Aprobada]") )

                # Entrenar

                if verbose:
                    printC('BLUE', "\nEntrenamiento. Empleando sesion A")
                vectorMediasA_C1        = mean( aplanado_SujetoA_SesionA_C1, axis=1 )
                vectorMediasB_C1        = mean( aplanado_SujetoB_SesionA_C1, axis=1 )
                vectorDesviacionesA_C1  = std ( aplanado_SujetoA_SesionA_C1, axis=1 )
                vectorDesviacionesB_C1  = std ( aplanado_SujetoB_SesionA_C1, axis=1 )

                if verbose:
                # Mostrar dimensiones de los vectores de entrenamiento en C1
                    print("Vectores medias C1\t- A:", shape( vectorMediasA_C1 ), "\tB:", shape( vectorMediasB_C1 ) )
                    print("Vectores desv. C1\t- A:", shape(vectorDesviacionesA_C1), "\tB:", shape(vectorDesviacionesB_C1))

                vectorMediasA_C2        = mean( aplanado_SujetoA_SesionA_C2, axis=1 )
                vectorMediasB_C2        = mean( aplanado_SujetoB_SesionA_C2, axis=1 )
                vectorDesviacionesA_C2  = std ( aplanado_SujetoA_SesionA_C2, axis=1 )
                vectorDesviacionesB_C2  = std ( aplanado_SujetoB_SesionA_C2, axis=1 )

                # Mostrar dimensiones de los vectores de entrenamiento en C2

                if verbose:
                    print("Vectores medias C2\t- A:", shape( vectorMediasA_C2 ), "\tB:", shape( vectorMediasB_C2 ) )
                    print("Vectores desv. C2\t- A:", shape(vectorDesviacionesA_C2), "\tB:", shape(vectorDesviacionesB_C2))

                aceptados = calcularAceptados( aplanado_SujetoA_SesionB_C1, vectorMediasB_C1, vectorDesviacionesB_C1,
                                               aplanado_SujetoA_SesionB_C2, vectorMediasB_C2, vectorDesviacionesB_C2, 0.72, 1.92E+00, 6.36E-01, verbose,
                                               4.37E+59, 1.95E+56, 2.04E+00, 6.55E-01)
                total = shape( aplanado_SujetoB_SesionB_C1 )[1]
                rechazados = total-aceptados

                print ( "Sujeto A: %i, Sujeto B: %i. Aceptado: %i (%.2f%%)\tRechazados: %i (%.2f%%)" % ( sujetoA, sujetoB, aceptados, aceptados * 100 / total, rechazados, rechazados * 100 / total )   )


# Llamada metodo principal
# main()
