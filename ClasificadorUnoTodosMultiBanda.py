# Clasificador bayesiano con 9 sujetos, mismo movimiento
# distincion uno contra todos
# Programa de honores UDLAP
from random         import shuffle, seed
from dit.other      import renyi_entropy
from numpy          import mean, std, median, shape, array, var, delete, reshape
from scipy.io       import loadmat
from scipy.signal   import butter, lfilter
from scipy.stats    import norm, skew, kurtosis
from MuestraDatos   import MAD, ajustarNumeroExperimentos, localizarIndicesConservar
from math           import inf
from MuestraDatos   import filtrar as filtrarOutliers
from dimFractal     import katz, hfd

"""
El metodo carga los datos de los sujetos dados en una clase determinada
Input:  sujetoA             = Valor entero con el numero de sujeto a cargar 
        sujetosB            = Lista de valores enteros con los indices restantes 
        claseUtilizada      = Clase que se distinguira en tipo str
Output: Tupla con los datos cargados ( S1(x,y,z), [S2(x,y,z),S3(x,y,z),..S9(x,y,z)] ) 
"""
def cargar( sujetoA, sujetosB, claseUtilizada ):

    # Carga del sujeto A
    sujetoA_nombreArchivo = "./Sujetos/S" + str( sujetoA )              # Nombre de la base de datos del sujetoA
    sujetoA = loadmat(sujetoA_nombreArchivo + ".mat")[claseUtilizada]   # Cargar datos del sujetoA

    # Cargar los demas sujetos (e.g. 2-9)
    sujetoB = []
    for sujeto in sujetosB:
        sujetoB_nombreArchivo = "./Sujetos/S" + str( sujeto )  # Nombre de la base de datos del sujetoB
        sujetoB.append( loadmat(sujetoB_nombreArchivo + ".mat")[claseUtilizada] ) # Cargar datos del sujetoB (n)

    # Devolver tupla. El primer elemento contiene los datos de un sujeto,
    # el segundo contiene datos de los sujetos restantes (e.g. 2-9)
    return ( sujetoA, sujetoB )


"""
El metodo aplica un filtro paso banda los datos de los sujetos
Input:  sujetos             = Tupla compuesta en la forma ( S1(x,y,z), [S2(x,y,z),S3(x,y,z),..S9(x,y,z)] ) 
        frecuenciaInferior  = Frecuencia inferior para el filtro paso banda
        frecuenciaSuperior  = Frecuencia superior para el filtro paso banda
        Fs                  = Frecuencia de muestreo         
Output: Datos filtrados en la misma composicion de entrada ( S1(x,y,z), [S2(x,y,z),S3(x,y,z),..S9(x,y,z)] ) 
"""
def filtrar ( sujetos, frecuenciaInferior, frecuenciaSuperior, Fs ):

    # Diseño del filtro
    b,a = butter(4, [frecuenciaInferior/(Fs/2),frecuenciaSuperior/(Fs/2)], btype='bandpass')

    # Filtrado de los datos
    sujetoA = lfilter(b,a,sujetos[0],1)     # Filtrar sujeto A. Clase 1. Dimensión 2 (indice 1)

    # Iterar sobre cada sujeto del conjunto y filtrarlo
    # Filtrar sujetos B. Clase 1. Dimensión 2 (indice 1)
    sujetosB = [ lfilter(b,a,sujeto,1) for sujeto in sujetos[1] ]

    return ( sujetoA, sujetosB )


"""
El metodo extrae una caracteristica de los datos proporcionados de los sujetos
Input:  datosSujetos  = Tupla compuesta en la forma con datos filtrados ( S1(x,y,z), [S2(x,y,z),S3(x,y,z),..S9(x,y,z)] )  
        funcion       = funcion utilizada para extraer las caracteristicas
        canales       = Canales en los que se opera 
Output: Tupla en la forma ( S1 (canales,exp) , [S2(canales,exp),S3(canales,exp),..S9(canales,exp)] )
"""
def extraerCaracteristicas ( datosSujetos, funcion, canales ):


    # Calcular el vector de caracteristicas mediante comprension de lista
    # Primero se obtiene la lista de las n_muestras. Se calcula la caracteristica de las n_muestras
    # El proceso se repite para generar n_experimentos listas
    # Se repite nuevamente por cada canal en la lista dada
    # Se concluye con una lista con shape (canales x caracteristicas )
    caracteristicas_SA = array([[funcion([ datosSujetos[0] [canal - 1][j][k] for j in range( shape( datosSujetos[0] )[1] )])
                                 for k in range( shape ( datosSujetos[0] )[2] )] for canal in canales])

    caracteristicas_SB = [ array([[funcion([ sujeto [canal - 1][j][k] for j in range( shape( sujeto )[1] )])
                                 for k in range( shape( sujeto )[2] )] for canal in canales]) for sujeto in datosSujetos[1] ]

    return ( caracteristicas_SA, caracteristicas_SB )



"""
El metodo remueve outliers de los datos proporcionados para todas las bandas de frecuencia
Input:  datosSujetoTodasBandas =    Tupla en la forma (bandas, canales, expermientos) 
                                    con los datos sin filtrado
                                    Los datos de cada sujeto son un numpy.ndarray
        desplazamiento =            numero de veces que valor del experimento (+/-) puede
                                    desviarse de la mediana sin ser eliminado.
                                    Un valor fuera de dicho umbral se considera outlier
Output: Datos en la misma forma de entrada con los datos filtrados 
"""
def removerOutliersSujeto ( datosSujetoTodasBandas, desplazamiento ):

    # Calcular numero de canales y experimentos
    _, n_canales, n_experimentos = shape( datosSujetoTodasBandas )

    # Inicializar con un set que contenga todos los indices de los datos a filtrar
    indicesOriginales = set( list( range( shape( datosSujetoTodasBandas )[2] ) ))
    indicesConservar = indicesOriginales.copy()

    for datos in datosSujetoTodasBandas:

        umbralCanales  = [ ( median(canal) - desplazamiento * MAD(canal),       # Calcular umbral entre el que se
                              median(canal) + desplazamiento * MAD(canal) )     # permitira pasar los datos
                          for canal in datos ]

        # Buscar indices a conservar
        # Calcular la interseccion de los indices
        indicesConservar = indicesConservar.intersection( localizarIndicesConservar( n_experimentos, n_canales, umbralCanales, datos.tolist( )  ) )

    # Calcular indices a eliminar con diferencia de conjuntos
    indicesEliminar = list ( indicesOriginales - indicesConservar )

    # Borrar indices eliminar del axis 2 (bandas, canales, exp)
    return delete( datosSujetoTodasBandas , indicesEliminar , axis=2 )


"""
El metodo remueve outliers de los datos proporcionados de los sujetos.
Por convencion se emplea el metodo de desviacion de la mediana absoluta (MAD)
Input:  datosSujetos =      Tupla en la forma (Banda 1 ( S1 (canales,exp) , 
                            [S2(canales,exp),S3(canales,exp),..S9(canales,exp)] ) ), ... Banda 2
                            con los datos sin filtrado
                            Los datos de cada sujeto son un numpy.ndarray
        desplazamiento =    numero de veces que valor del experimento (+/-) puede
                            desviarse de la mediana sin ser eliminado.
                            Un valor fuera de dicho umbral se considera outlier
Output: Datos en la forma (S1 (bandas, canales, exp), [S2 (bandas, canales, exp),S2 (bandas, canales, exp), ... ]
"""
def removerOutliers ( datosSujetos, desplazamiento ):

    # Convertir en la forma (bandas, canales, experimentos) para SA (Ej. S1)
    datosSA = array ( [ banda[0] for banda in datosSujetos ] )

    # Aplicar filtro
    caracteristicas_SA_Filtradas = removerOutliersSujeto( datosSA, desplazamiento )

    # Mostrar resultado de filtro en SA
    formaOriginal = shape(datosSA)
    formaSalida = shape(caracteristicas_SA_Filtradas)
    expOriginales = formaOriginal[2]
    expSalida = formaSalida[2]
    print("SA Original:", shape(datosSA), "Salida:", formaSalida,
          "Perdida:%.2f%%" % ((1 - (expSalida / (expOriginales))) * 100))

    # Lista para guardar los datos filtrados [ e.g. S2, S3, ...]
    caracteristicas_SB_Filtradas = []

    # Iterar sobre cada sujeto en la lista B [ e.g. S2, S3, ...]
    for sujeto in range( len ( datosSujetos[0][1] ) ):

        # Convertir en la forma (bandas, canales, experimentos) para SB (Ej. S2, S3,...)
        datosSB             = array( [ banda[1][sujeto] for banda in datosSujetos] )
        datosSinOutliersSB  = removerOutliersSujeto( datosSB, desplazamiento )
        caracteristicas_SB_Filtradas.append( datosSinOutliersSB )

        # Calcular y desplegar perdida
        formaOriginal = shape( datosSB )
        formaSalida =   shape (datosSinOutliersSB)
        expOriginales = formaOriginal[2]
        expSalida = formaSalida[2]
        print( "SB Original:", formaOriginal, "Salida:", formaSalida , "Perdida:%.2f%%" % ( ( 1 - (expSalida / ( expOriginales ) ) ) * 100 ) )


    return ( caracteristicas_SA_Filtradas, caracteristicas_SB_Filtradas )


def aplanarDimensionBandasFrecuenciaIndividual( datos ):

    return reshape( datos, (shape(datos)[0] * shape(datos)[1], shape(datos)[2]) ).tolist()

def aplanarDimensionBandasFrecuencia( datos ):

    # Separar sujeto A y B
    datosSA = datos[0]
    datosSB = datos[1]

    # Invocar metodo auxiliar para aplanar los datos de cada sujeto
    return ( aplanarDimensionBandasFrecuenciaIndividual( datosSA ),
    [ aplanarDimensionBandasFrecuenciaIndividual ( sujeto ) for sujeto in datosSB ] )

"""
El metodo normaliza el numero de experimentos de un conjunto de sujetos 
al minimo entre ellos
Input:  datosSujetos  = Tupla en la forma ( S1 (canales,exp) , [S2(canales,exp),S3(canales,exp),..S9(canales,exp)] ) 
Output: Ninguno**
"""
def normalizarNumeroExperimentos ( datosSujetos ):

    # Calcular minimo
    longitudes = [ len( datosSujetos[0][0] ) ]
    longitudes += [ len( sujeto[0] ) for sujeto in datosSujetos[1] ]
    minimo = min ( longitudes )

    # Invocar al metodo de reduccion para sujetoA y sujetos en la lista B
    ajustarNumeroExperimentos( datosSujetos[0], minimo )
    [  ajustarNumeroExperimentos( sujeto, minimo ) for sujeto in datosSujetos[1] ]


"""
El metodo recibe un conjunto de datos en la forma [canales, exp] y un porcentaje de 
entrenamiento. Se devuelven los datos en una tupla ( entrenamiento [ canales, exp], clasificacion[canales, exp] )
Input:  conjuntoDatos       =   Lista en la forma [ canales, exp ]
        porcentajeEntrenar  =   float con el porcentaje de datos que se destinaran
                                al entrenamiento (0.5 = 50%)
Output: tupla en la forma ( entrenamiento [ canales, exp], clasificacion[canales, exp] )
"""
def distribuirDatosEntrenamientoYClasificacion ( conjuntoDatos, porcentajeEntrenar ):

    # Obtener el punto en que se dividira (segun el porcentaje de datos para entrenar)
    punto_corte = int( shape( conjuntoDatos )[1] * porcentajeEntrenar)

    # Efectuar division y devolver tupla
    return ( array(conjuntoDatos)[:,:punto_corte].tolist(), array(conjuntoDatos)[:,punto_corte:].tolist() )



"""
El metodo recibe un conjunto de datos en la forma [canales, exp] y una semilla.
Los datos se mezclan aleatoriamente en cada canal
Input:  datos  =    Lista en la forma [ canales, exp ]
Output: misma forma de entrada [ canales, exp ] con los datos reordenados
"""
def mezclarDatos ( datos ):

    [shuffle(canal) for canal in datos]     # Ordenar de forma aleatoria los datos del sujeto A


"""
El metodo concatena un grupo de experimentos (de varios sujetos) en uno solo
Nota: no se aplia reduccion. Si la entrada es [S1,S2,..S9]. La salida tendra 
S1 + S2 +,... + S9 experimentos
Input:  conjuntoExperimentos  = Lista en la forma [S2(canales,exp),S3(canales,exp),..S9(canales,exp)] 
Output: experimentosConcatenados = [ canales, exp ]
"""
def concatenarExperimentos ( conjuntoExperimentos ):

    numCanales = shape( conjuntoExperimentos )[1]                   # Calcular numero de canales
    experimentosConcatenados = [ [] for i in range( numCanales ) ]  # Inicializar lista para guardar concatenacion

    # Iterar sobre cada sujeto de la lista dada
    for sujeto in conjuntoExperimentos:

        # Iterar sobre cada canal y unirlo a la lista correspondiente en experimentosConcatenados
        for numCanal in range ( numCanales ):
            experimentosConcatenados[ numCanal ] += sujeto[ numCanal ]

    return experimentosConcatenados


"""
El metodo clasifica un conjunto de datos dada una informacion de entrenamiento
Input:  probabilidadA=  Probabilidad de ocurrencia de la clase A
        probabilidadB=  Probabilidad de ocurrencia de la clase B
        canales=        Canales en los cuales se debe operar
        medias_SA=      Lista de medias de la clase A (Entrenamiento)
        medias_SB=      Lista de medias de la clase B (Entrenamiento)
        medias_SA=      Lista de desviaciones de la clase A (Entrenamiento)
        medias_SA=      Lista de desviaciones de la clase B (Entrenamiento)
        tipoCorrecto=   Variable string que indica a que tipo pertenecen los datos proporcionados 'A' o 'B'
                        usada para contar aciertos
        datosClasificar=coleccion de datos por clasificar (canales, exp)
Output: Lista de tuplas ( clasificados A, clasificados B, aciertos ) con cada sujeto.
"""
def clasificar( probabilidadA, probabilidadB, canales,
                medias_SA, medias_SB, desviacion_SA, desviacion_SB, tipoCorrecto,
                datosClasificar ):

    # Contadores de clasificados como sujeto A y B y aciertos
    conteo_SA = conteo_SB = aciertos = 0

    # Por cada experimento
    for numExp in range( shape( datosClasificar )[1] ):

        # Calcular la probabilidad de pertenencia a cada clase
        P_SA = probabilidadA
        P_SB = probabilidadB

        # Por cada canal
        for canal in canales:
            # Multiplicar las probabilidades por P(x) donde x es la varianza del experimento n
            P_SA *= norm.pdf(datosClasificar[canal - 1][numExp], medias_SA[canal - 1], desviacion_SA[canal - 1])
            P_SB *= norm.pdf(datosClasificar[canal - 1][numExp], medias_SB[canal - 1], desviacion_SB[canal - 1])

        # Ver en que sujeto la probabilidad es mas alta
        if P_SA > P_SB:
            conteo_SA += 1
            aciertos += 1 * (tipoCorrecto == 'A')
        else:
            conteo_SB += 1
            aciertos += 1 * (tipoCorrecto == 'B')

    return ( conteo_SA, conteo_SB, aciertos )


"""
El metodo recibe una lista de resultados obtenidas a lo largo de multiples iteraciones
y se encarga de promediarlos y generar una sola tupla con clasificados como A, 
clasificados como B y aciertos
Input:  resultados = [ (A, B, aciertos) ]
"""
def unirResultadosIteraciones ( resultados ):
    clasificadosA = clasificadosB = aciertos = 0
    for tupla in resultados:
        clasificadosA += tupla[0]
        clasificadosB += tupla[1]
        aciertos += tupla[2]
    return ( clasificadosA  / len ( resultados ),
             clasificadosB  / len ( resultados ),
             aciertos       / len ( resultados) )

"""
El metodo entrena al clasificador, efectua la clasificacion
y evalua la efectividad empleando el metodo de validacionCruzada.
La efectividad obtenida es impresa
Input:  datos = coleccion de datos sin outliers de sujetoA y sujetosB
        porcentajeEntrenar =    Porcentaje de los datos que se emplearan para 
                                entrenar al clasificador 
        iteraciones =           Numero de iteraciones 
        semilla =               Semilla empleada para mezclar los datos
Output: Lista de tuplas ( clasificados A, clasificados B, aciertos ) con cada sujeto.
        Nota. Son clasificaciones promedio
"""
def validacionCruzada ( datos, porcentajeEntrenar, canales, iteraciones, semilla, forzarEquiprobabilidad ):

    # Establecer la semilla del generador aleatorio
    seed(semilla)

    # Lista para almacenar los resultados de cada iteracion
    resultadosPorIteracion = [ [] for sujeto in datos[1] ]
    resultadosPorIteracion.append([])

    composicionDatosYaMostrada = False

    # Realizar las iteraciones solicitadas
    for numIteracion in range ( iteraciones ):

        # ************** Fase de entrenamiento **************

        # Mezclar datos para clase A (e.g. S1)
        mezclarDatos( datos[0] )

        # Mezclar datos para cada elemento de la clase B (e.g. S2, S3, ... S9)
        [ mezclarDatos( sujeto ) for sujeto in datos[1] ]

        # Dividir datos de la clase A (e.g. S1)
        entrenamientoA, clasificacionA = distribuirDatosEntrenamientoYClasificacion( datos[0], porcentajeEntrenar )

        # Para dividir datos de las clases B (e.g. S2, S3, ... S9)
        distribucionB = []

        # Si es necesario forzar la equiprobabilidad
        if forzarEquiprobabilidad:
            distribucionB = [ distribuirDatosEntrenamientoYClasificacion( sujeto, porcentajeEntrenar/len( datos[1] ) )
            for sujeto in datos[1] ]

        else:
            distribucionB = [distribuirDatosEntrenamientoYClasificacion(sujeto, porcentajeEntrenar) for sujeto in
                             datos[1]]

        clasificacionB = array(distribucionB)[:, 1, :].tolist()

        # Concatenar datos para entrenamiento de la clase B (e.g. S2, S3, ... S9)
        entrenamientoB= concatenarExperimentos( array(distribucionB)[:,0,:].tolist() )


        # Entrenar (calcular media y desviacion de cada clase)
        medias_SA     = mean( entrenamientoA, axis=1 )
        medias_SB     = mean( entrenamientoB, axis=1 )
        desviacion_SA = std ( entrenamientoA, axis=1 )
        desviacion_SB = std ( entrenamientoB, axis=1 )

        # ************** Fase de clasificacion **************

        # Probabilidad de cada clase
        P_SA = shape ( entrenamientoA )[1] / ( shape ( entrenamientoA )[1] + shape ( entrenamientoB )[1])
        P_SB = shape(entrenamientoB)[1] / (shape(entrenamientoA)[1] + shape(entrenamientoB)[1])

        # Mostrar condiciones del clasificador
        if ( not composicionDatosYaMostrada ):
            print("\nMedia SA:", medias_SA)
            print("Media SB:", medias_SB)
            print("Desviacion SA:", desviacion_SA)
            print("Desviacion SB:", desviacion_SB)
            print("A. Datos entrenamiento:", shape(entrenamientoA), "Datos clasificacion:", shape(clasificacionA))
            print ("B. Datos entrenamiento:", shape( entrenamientoB ), "Datos clasificacion:", shape( clasificacionB ) )
            print("Probabilidad A inicial: %.2f Probabilidad B inicial: %.2f" %( P_SA, P_SB ) )
            composicionDatosYaMostrada = True

        # Clasificar experimentos que se sabia eran A (e.g. S1)
        resultadosPorIteracion[0].append(
            clasificar( P_SA, P_SB, canales, medias_SA, medias_SB, desviacion_SA, desviacion_SB, 'A', clasificacionA) )

        # Clasificar experimentos que se sabe son B (e.g. S2, S3, ... S9)
        for sujeto in range ( shape( clasificacionB )[0] ):
            resultadosPorIteracion[ sujeto + 1 ].append(
                clasificar(P_SA, P_SB, canales, medias_SA, medias_SB, desviacion_SA, desviacion_SB, 'B', clasificacionB[ sujeto ] ) )

    return [ unirResultadosIteraciones( resultado ) for resultado in resultadosPorIteracion ]



"""
El metodo se encarga de realizar la tarea completa de clasificacion
Input:  numeroSA                    Numero de sujeto que se distinguira contra el resto
        movimiento                  Movimiento comun "C1" o "C2"
        freqInfFiltro               Frecuencia inferior para el filtro paso banda
        freqSupFiltro               Frecuencia superior para el filtro paso banda
        Fs                          Frecuencia de muestreo
        funcionCaracteristicas      Funcion para extraer caracteristicas
        desplazamientoMAD           Desplazamiento maximo permitido de la mediana para no ser outlier
        porcentajeDatosEntrenar     Porcentaje de datos usados para entrenar
        canalesUtilizar             Lista de canales a considerar en la tarea de clasificacion
        iteraciones                 Iteraciones de la validacion cruzada
        semilla                     Semilla empleada para el generador random
"""
def ejecutarClasificador( numeroSA, movimiento, frecuenciasFiltrado, Fs, funcionCaracteristicas, desplazamientoMAD,
                          porcentajeDatosEntrenar, canalesUtilizar, iteraciones, semilla, forzarEquiprobabilidad ):

    # Calcular los sujetos restantes
    numerosSB = list ( range (1,10) )
    numerosSB.remove( numeroSA )

    # Cargar y mostrar los sujetos cargados
    datosSujetos = cargar( numeroSA, numerosSB, movimiento )
    print("SujetoA:", numeroSA, " SujetosB:", numerosSB )
    print("SujetoA:", shape( datosSujetos[0] ), " SujetosB:", [ shape( sujeto ) for sujeto in datosSujetos[1] ] )

    # Mostrar la caracteristica que sera extraida
    print("\nCaracteristica:", funcionCaracteristicas.__name__)

    # Almacenar las caracteristicas en todas las bandas de frecuencia
    caracteristicas = [ ]

    for freqInfFiltro, freqSupFiltro in frecuenciasFiltrado:

        # Filtrar datos con paso banda
        datosFiltrados = filtrar( datosSujetos, freqInfFiltro, freqSupFiltro, Fs )
        print( "Filtrado entre - Inferior: ", freqInfFiltro, "\tSuperior:", freqSupFiltro )

        # Extraccion de caracteristicas
        caracteristicas.append( extraerCaracteristicas( datosFiltrados, funcionCaracteristicas, canalesUtilizar ) )

    # Remover outliers (MAD)
    print("\nRemocion outliers")
    datosSinOutliers =  removerOutliers( array( caracteristicas ) , desplazamientoMAD )

    # Aplanar datos (pasar varias bandas como canales extras)
    datosSinOutliers = aplanarDimensionBandasFrecuencia( datosSinOutliers )

    # Normalizacion en el numero de experimentos
    print("\nNormalizacion en numero de experimentos")
    normalizarNumeroExperimentos( datosSinOutliers )
    print("SujetoA:", shape( datosSinOutliers[0] ), " SujetosB:", [ shape(sujeto) for sujeto in datosSinOutliers[1] ] )

    # Tarea de entrenamiento y clasificacion
    canalesUtilizar = list ( range (1, shape( datosSinOutliers[0] )[0] + 1) )

    resultados = validacionCruzada( datosSinOutliers, porcentajeDatosEntrenar, canalesUtilizar, iteraciones, semilla, forzarEquiprobabilidad )

    # Despliegue de resultados
    print( "\nResultado sujeto aislado" )

    print ("Sujeto %d (A) -> A: %.2f\tB: %.2f\tAciertos: %.2f\tEfectividad: %.2f%%" %
           (numeroSA, resultados[0][0], resultados[0][1], resultados[0][2],
            resultados[0][2] * 100 / (resultados[0][0] + resultados[0][1])) )

    print( "Resultados sujetos concatenados" )
    for i in range( len (numerosSB) ):
        print("Sujeto %d (B) -> A: %.2f\tB: %.2f\tAciertos: %.2f\tEfectividad: %.2f%%" %
              (numerosSB[i], resultados[i + 1 ][0], resultados[i + 1 ][1], resultados[i + 1][2],
               resultados[i + 1][2] * 100 / (resultados[i + 1][0] + resultados[i + 1][1])))

"""
Funcion principal. 
El metodo se encarga de la definicion de los parametros del 
clasificador y de la ignicion del mismo
"""
def main():

    # Parametros de la clasificacion
    sujetoAislado       = 1                     # Sujeto (e.g. 1) que se debe distinguir contra el resto (e.g. S2, S3,...S9)
    canales             = [1,2,3]               # Canales a considerar
    desplazamiento      = inf                   # Veces (MAD) distanciamiento de la mediana
    Fs                  = 250                   # Frecuencia de muestreo
    porcentajeEntrenar  = 0.5                   # Porcentaje de los datos que se usaran para entrenar
    iteraciones         = 30                    # Iteraciones a realizar
    semilla             = 4                     # Semilla para mezclar los datos
    claseUtilizada      = 'C1'                  # Clase a utilizar para distinguir entre sujetos
    frecuenciasFiltro   = [ (36,43),
                            (4 ,8),
                            (47,48),
                            (11,15)]             # Frecuencia inferior para filtro paso banda
    funcionCaracteristicas = std                # Funcion para la extraccion de caracteristicas
    forzarEquiprobabilidad = False              # Si se ajusta el numero de experimentos de entrenamiento
                                                # en sujetos B para obtener equiprobabilidad entre A y B

    ejecutarClasificador( sujetoAislado,
                          claseUtilizada,
                          frecuenciasFiltro,
                          Fs,
                          funcionCaracteristicas,
                          desplazamiento,
                          porcentajeEntrenar,
                          canales,
                          iteraciones,
                          semilla,
                          forzarEquiprobabilidad)
