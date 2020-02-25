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
    
    # Balancear la cantidad de datos de entrenamiento
    n_experimentos = min(n_experimentosA, n_experimentosB)
    
    # Calcular las varianzas mediante comprension de lista 
    # Primero se obtiene la lista de las n_muestras. Se calcula la varianza de las n_muestras
    # El proceso se repite para generar n_experimentos listas
    # Se repite nuevamente por cada canal en la lista dada 
    # Se concluye con una lista con shape (canales x varianzas)
    varianzas_SA = [[ var([sujetoA[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentos) ] for canal in canales]
    varianzas_SB = [[ var([sujetoB[canal-1][j][k] for j in range(n_muestras)]) for k in range(n_experimentos) ] for canal in canales]

    # Obtener el punto en que se dividira (segun el porcentaje de datos para entrenar)
    punto_corte = int(n_experimentos * porcentajeEntrenar)

    # Datos clasificados (Calcular cuantos datos se destinan a clasificacion)
    datosTest = n_experimentos-punto_corte

    K_resultados    = []    # Guardar las efectividades de cada iteracion del K-Fold

    # Iterar K veces
    for K_experimento in range(iteraciones):

        # Mezclar datos 
        seed(semillaKFold)          # Estandarizar la semilla
        shuffle(varianzas_SA)       # Ordenar de forma aleatoria los datos del sujeto A

        seed(semillaKFold)          # Estandarizar la semilla
        shuffle(varianzas_SB)       # Ordenar de forma aleatoria los datos del sujeto B

        # Delimitar datos designados a entrenamiento y extraerles media y desviacion
        #  0 0 0 0 0 0 0  [1 1 1 1 1 1 1]

        medias_SA     = [mean(canal_varianza[0:punto_corte]) for canal_varianza in varianzas_SA]   
        medias_SB     = [mean(canal_varianza[0:punto_corte]) for canal_varianza in varianzas_SB]
        desviacion_SA = [std (canal_varianza[0:punto_corte]) for canal_varianza in varianzas_SA]
        desviacion_SB = [std (canal_varianza[0:punto_corte]) for canal_varianza in varianzas_SB]

        # Fase de clasificacion------------------------------------------------------------------------------
        # Delimitar datos de clasificacion (test)
        
        varianzas_test_SA = [canal_varianza[punto_corte:] for canal_varianza in varianzas_SA]
        varianzas_test_SB = [canal_varianza[punto_corte:] for canal_varianza in varianzas_SB]

        # Contadores de clasificados como sujeto A y B y aciertos
        conteo_SA = conteo_SB = aciertos = 0        

        # Clasificar experimentos que se sabia eran S1
        # Por cada experimento
        for exp in range(datosTest):

            # Probabilidad de cada clase
            P_SA = P_SB = 0.5

            # Por cada canal
            for canal in range(len(canales)):

                # Multiplicar las probabilidades por P(x) donde x es la varianza del experimento n
                P_SA *= norm.pdf(varianzas_test_SA[canal-1][exp],medias_SA[canal-1],desviacion_SA[canal-1])
                P_SB *= norm.pdf(varianzas_test_SA[canal-1][exp],medias_SB[canal-1],desviacion_SB[canal-1])


            # Ver en que sujeto la probabilidad es mas alta
            if P_SA > P_SB:
                conteo_SA +=1
                aciertos+=1
            else:
                conteo_SB +=1

        # Clasificar experimentos que se sabia eran S2
        for exp in range(datosTest):
            P_SA = P_SB = 0.5
            for canal in range(len(canales)):
                P_SA *= norm.pdf(varianzas_test_SB[canal-1][exp],medias_SA[canal-1],desviacion_SA[canal-1])
                P_SB *= norm.pdf(varianzas_test_SB[canal-1][exp],medias_SB[canal-1],desviacion_SB[canal-1])

            # Ver en que sujeto la probabilidad es mas alta 
            if P_SA > P_SB:
                conteo_SA +=1 
            else:
                conteo_SB +=1
                aciertos+=1
        
        K_resultados.append((conteo_SA,conteo_SB,aciertos))  # Guardar la efectividad K
        

    #print("Clasificados como sujeto A: ",[i[0] for i in K_resultados])
    #print("Clasificados como sujeto B: ",[i[1] for i in K_resultados])
    #print("Aciertos:                   ",[i[2] for i in K_resultados])
    #print("Efectividades:              ",[str(100*i[2]/(datosTest*2))+"%" for i in K_resultados])

    efectividadesPromedio=100*mean([i[2] for i in K_resultados])/(datosTest*2)
    #print("Efectividad promedio:       ",efectividadesPromedio,"%")
    return efectividadesPromedio

for sujeto_A in range(1,9):
    for sujeto_B in range(sujeto_A+1,10):
        for clase in range(1,3):

            # Parametros de los archivos
            sujetoA_nombreArchivo = "./Sujetos/S"+str(sujeto_A)    # Nombre de la base de datos del sujetoA
            sujetoB_nombreArchivo = "./Sujetos/S"+str(sujeto_B)    # Nombre de la base de datos del sujetoB

            efectividades ={}
            procesados = 0

            for bInf in range (1,49):
                for bSup in range (bInf,50):

                    # Parametros de la clasificacion
                    canales             = [1,2,3]               # Canales a considerar
                    Fs                  = 250                   # Frecuencia de muestreo
                    porcentajeEntrenar  = 0.5                   # Porcentaje de los datos que se usaran para entrenar
                    iteraciones         = 30                    # Iteraciones a realizar
                    semillaKFold        = 1                     # Semilla para mezclar los datos 
                    claseUtilizada      = 'C'+str(clase)        # Clase a utilizar para distinguir entre sujetos
                    bandaInferiorFiltro = bInf                  # Banda inferior de frecuencias a filtrar
                    bandaSuperiorFiltro = bSup                  # Banda superior de frecuencias a filtrar

                    efectividades[clasificar()]={"bInf":bInf,"bSup":bSup}

            print("SA:",sujeto_A,"SB:",sujeto_B,"C:",clase,"Efect:",max(efectividades),"Param:",efectividades[max(efectividades)])
