# Clasificador bayesiano con 9 sujetos, mismo movimiento
# Programa de honores UDLAP
from numpy        import var, mean, std, shape, zeros, prod
from scipy.signal import butter, lfilter, resample_poly
from scipy.io     import loadmat
from scipy.stats  import norm
from random       import shuffle, seed

def clasificar():
    
    #Fase de entrenamiento------------------------------------------------------------------------------
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
    varianzas_SA = [[ var([sujetoA[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentosA) ] for canal in canales]
    varianzas_SB = [[ var([sujetoB[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentosB) ] for canal in canales]

    # K-Fold cross validation
    seed(semillaKFold)          # Estandarizar la semilla
    shuffle(varianzas_SA)       # Ordenar de forma aleatoria los datos del sujeto A
    
    seed(semillaKFold)          # Estandarizar la semilla
    shuffle(varianzas_SB)       # Ordenar de forma aleatoria los datos del sujeto B

    # Determinar si la cantidad de datos de entrenamiento permite el K-Fold validation
    # Si no, lanzar excepcion
    if cantDatosEntrenar % K != 0:
        raise ValueError("El numero de datos a entrenar no es multiplo de K")

    
    K_resultados    = []    # Guardar las efectividades de cada iteracion del K-Fold
    inicioExp       = 0     # Guardar el inicio de la porcion que se considerara en el experimento K

    # Iterar K veces
    for K_experimento in range(K):

        # Calcular por cada canal, la media y desviacion estandar hasta los primeros n_experimentos
        medias_SA     = [mean(canal_varianza[inicioExp:inicioExp+n_experimentos]) for canal_varianza in varianzas_SA]   
        medias_SB     = [mean(canal_varianza[inicioExp:inicioExp+n_experimentos]) for canal_varianza in varianzas_SB]
        desviacion_SA = [std(canal_varianza[inicioExp:inicioExp+n_experimentos])  for canal_varianza in varianzas_SA]
        desviacion_SB = [std(canal_varianza[inicioExp:inicioExp+n_experimentos])  for canal_varianza in varianzas_SB]

        # Fase de clasificacion------------------------------------------------------------------------------
        # Contadores de clasificados como sujeto A y B y aciertos
        conteo_SA = conteo_SB = aciertos = 0

        # Clasificar experimentos que se sabia eran S1
        # Por cada experimento
        for exp in range(n_experimentosA):
            P_SA = P_SB = 0.5

            # Por cada canal
            for canal in range(len(canales)):

                # Multiplicar las probabilidades por P(x) donde x es la varianza del experimento n
                P_SA *= norm.pdf(varianzas_SA[canal-1][exp],medias_SA[canal-1],desviacion_SA[canal-1])
                P_SB *= norm.pdf(varianzas_SA[canal-1][exp],medias_SB[canal-1],desviacion_SB[canal-1])


            # Ver en que sujeto la probabilidad es mas alta
            if P_SA > P_SB:
                conteo_SA +=1
                aciertos+=1
            else:
                conteo_SB +=1

        # Clasificar experimentos que se sabia eran S2
        for exp in range(n_experimentosB):
            P_SA = P_SB = 0.5
            for canal in range(len(canales)):
                P_SA *= norm.pdf(varianzas_SB[canal-1][exp],medias_SA[canal-1],desviacion_SA[canal-1])
                P_SB *= norm.pdf(varianzas_SB[canal-1][exp],medias_SB[canal-1],desviacion_SB[canal-1])

            # Ver en que sujeto la probabilidad es mas alta 
            if P_SA > P_SB:
                conteo_SA +=1 
            else:
                conteo_SB +=1
                aciertos+=1
        
        K_resultados.append((conteo_SA,conteo_SB,aciertos))  # Guardar la efectividad K
        inicioExp = inicioExp + n_experimentos                  # Avanzar porcion de datos considerada en el K-Fold
        

    print("Clasificados como sujeto A: ",[i[0] for i in K_resultados])
    print("Clasificados como sujeto B: ",[i[1] for i in K_resultados])
    print("Aciertos:                   ",[i[2] for i in K_resultados])
    print("Efectividades:              ",[str(100*i[2]/(n_experimentosA+n_experimentosB))+"%" for i in K_resultados])

    efectividadesPromedio=100*mean([i[2] for i in K_resultados])/(n_experimentosA+n_experimentosB)
    print("Efectividad promedio:       ",efectividadesPromedio,"%")


# Parametros de los archivos
sujetoA_nombreArchivo = "./Sujetos/S1"    # Nombre de la base de datos del sujetoA
sujetoB_nombreArchivo = "./Sujetos/S2"    # Nombre de la base de datos del sujetoB


# Parametros de la clasificacion
canales             = [1,2,3]   # Canales a considerar
Fs                  = 250       # Frecuencia de muestreo
K                   = 3         # Numero K-fold
semillaKFold        = 1         # Semilla para mezclar arreglo en K-Fold
cantDatosEntrenar   = 45       # Cantidad de muestras a usar en el entrenamiento   
claseUtilizada      = 'C2'      # Clase a utilizar para distinguir entre sujetos
bandaInferiorFiltro = 20        # Banda inferior de frecuencias a filtrar
bandaSuperiorFiltro = 35        # Banda superior de frecuencias a filtrar

clasificar()
