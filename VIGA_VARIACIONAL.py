#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  4 18:24:19 2021

@author: FMagnani
Edited by: Angel Vázquez-Patiño

Editado por Pablo Molineros y Karen Sacoto
"""

import numpy as np
import tensorflow as tf


from VARIACIONAL_PINN import VIGA_VARIACIONAL_PINN
#from SC_plotting import plot_results, plot_error

import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import time



#%% main

if __name__ == "__main__":  
    
    # Semilla aleatoria
    np.random.seed(100)
    tf.random.set_seed(100)
    
    #Conteo para comparación de tiempo
    start = time.time()
    

#%% 
    ##############################
    ##   PREPARACION DE DATOS   ##
    ##############################
    
    num_train_samples = 181 # Número de puntos de colocación

    
    #Características geométricas y mecanicas de la viga
    
    q = 60000    #N/m
    L = 2.7      #m    
    E = 200e9     #Pa                         
    I = 0.000038929334  #m^4
    
    # Domain bounds
    lb = np.array([ 0.0]) # Limite inferior
    wb = np.array([   L]) # Limite superior
    
   
    
#%% 
    ###############################################
    ##   IMPOSICION DE CONDICIONES DE CONTORNO   ##
    ###############################################

    
    #Problema 1: Simplemente apoyada
    
    x_0 = np.zeros((1, 1)) # Posición de contorno u(x_0) = 0
    x_1 = np.linspace(0, L, num_train_samples).reshape((-1,1)) # distribución de puntos de colocación para el entrenamiento
    x_2 = L*np.ones((1, 1)) # Posición de contorno u(x_2) = 0
    

    w_0 = 0*np.ones((1, 1)) # Condición de contorno u(0) = w_2
    w_2 = 0*np.ones((1, 1)) # Condición de contorno u(L) = w_2
    
    x_3 = L/2*np.ones((1, 1))
    w_3 = 0*np.ones((1, 1))
   
    """  
    #Problema 2: Empotrada - libre
    
    x_0 = np.zeros((1, 1)) # Posición de contorno u(x_0)
    x_1 = np.linspace(0, L, num_train_samples).reshape((-1,1)) # distribución de puntos de colocación para el entrenamiento
    x_2 = np.zeros((1, 1)) # Posición de contorno u(x_2)

    w_0 = 0*np.ones((1, 1)) # Condición de contorno u(0) = w_0
    w_2 = 0*np.ones((1, 1)) # Condición de contorno w_x(0) = w_2
    
    
    w_3 = 0*np.ones((1, 1))
    x_3 = L/2*np.ones((1, 1))
    
    """
    """
     #Problema 3: Empotrada - simplemente apoyada
    
    x_0 = np.zeros((1, 1)) # Posición de contorno u(x_0)
    x_1 = np.linspace(0, L, num_train_samples).reshape((-1,1)) # puntos de colocación para el entrenamiento
    x_2 = np.zeros((1, 1)) # Posición de contorno u(x_2)
    x_3 = L*np.ones((1, 1)) # Posición de contorno u(x_3)

    w_0 = 0*np.ones((1, 1)) # Condición de contorno u(0) = w_0
    w_2 = 0*np.ones((1, 1)) # Condición de contorno w_x(0) = w_2
    w_3 = 0*np.ones((1, 1)) # Condición de contorno u(L) = w_3
 
    """  

    
    #Conversión de a tensores para la red neuronal
    
    x0 = tf.convert_to_tensor(x_0[:, 0])
    x1 = tf.convert_to_tensor(x_1[:, 0])
    x2 = tf.convert_to_tensor(x_2[:, 0])
    x3 = tf.convert_to_tensor(x_3[:, 0])
    
    w0 = tf.convert_to_tensor(w_0[:, 0])
    w2 = tf.convert_to_tensor(w_2[:, 0])
    w3 = tf.convert_to_tensor(w_3[:, 0])

 
    #Creación de capas y neuronas para la red
    layers = [1,20,20,1]
    
    
    #Entrenamiento de la red neuronal informada por la física
    model = VIGA_VARIACIONAL_PINN(x0, w0, x1, x2, w2, x3, w3, wb, lb, layers)


    
#%%

    ###############################################
    ##   ENTRENAMIENTO Y PREDICCION DEL MODELO   ##
    ###############################################

    adam_iterations = 500  # Número de iteraciones de Adam
    lbfgs_max_iterations = 1000 # Máximas iterationes de lbfgs

    # Entrenamiento
    Adam_hist,LBFGS_hist = model.train(adam_iterations, lbfgs_max_iterations)


    #Espacio para la predicción en el dominio requerido con la cantidad de N puntos de colocación
    x = np.linspace(0, L, num_train_samples).reshape((-1,1))
    x = tf.convert_to_tensor(x[:, 0])
    
    
    #Resultado de pasar el espacio por la red neuronal
    res, res_derivada = model.predict(x)
       
    end = time.time()
    print("Tiempo de entrenamiento total: {res:.2f}".format(res=end - start)," s")
    


#%%

    #############################
    ##   GRAFICAS DEL MODELO   ##
    #############################



    ### Problema 1: Simplemente apoyada
    titulo = "PROBLEMA 1: VIGA SIMPLEMENTE APOYADA"
    analitica = q/(24*E*I)*(-x**4 + 2*L*x**3 - L**3*x)
    
    flecha = 5*q*L**4/(384*E*I)
    flecha_aprox = res[len(res)//2]
    
    posx1,posy1 = 0.15,0.2
    
    posx2,posy2 = 0.65,0.2
   
    
   
    """      
    ### Problema 2: Empotrada - libre
    titulo="PROBLEMA 2: VIGA EMPOTRADA - LIBRE"
    analitica = q/(24*E*I)*(-x**4 + 4*L*x**3 - 6*L**2*x**2)
    
    flecha = q/(24*E*I)*(-x[-1]**4 + 4*L*x[-1]**3 - 6*L**2*x[-1]**2)
    flecha_aprox = res[-1]
    
    posx1,posy1 = 0.15,0.2
    
    posx2,posy2 = 0.15,0.35
    """ 
    
    """ 
      
    ### Problema 3: Empotrada - simplemente apoyada
    titulo="PROBLEMA 3: VIGA EMPOTRADA - SIMPLEMENTE APOYADA"
    analitica = q/(24*E*I)*(-x**4 + 5/2*L*x**3 - 3/2*L**2*x**2)
    
    x=-L*(33**(1/2)-15)/16
    flecha = q/(24*E*I)*(-x**4 + 5/2*L*x**3 - 3/2*L**2*x**2)
    flecha_aprox = res[min(range(len(x)), key=lambda i: abs(x[i]-x))]
    
    posx1,posy1 = 0.45,0.8
    posx2,posy2 = 0.71,0.18
    """   

 

    #PLOT 1: PREDICCION PINN VS SOLUCION EXACTA
    
    plt.style.use('bmh')

    fig = plt.figure(dpi = 300)
    plt.rcParams["font.family"] = "Arial"

    plt.title(titulo,fontweight="bold")
    plt.gca().invert_yaxis()
    plt.plot(x, -analitica, ls='-', color='k', label='Solución exacta',linewidth=1)
    plt.plot(x, res, ls='--', color='b', label='Solución aproximada',linewidth=1)
    

    plt.figtext(posx1,posy1,"$MSE: {res:.2e}$".format(res=mean_squared_error(res, -analitica)),
                 wrap = True, fontsize = 10, 
            bbox ={'facecolor':'mediumpurple', 'alpha':.5, 'pad':5})
       

    roots = [flecha_aprox.numpy()[0].tolist()]


    mark = res.numpy().tolist().index(roots)


    plt.plot(x[mark],res[mark], ls="", marker="o", label="Deflexión máxima")
    
   
    plt.figtext(posx2,posy2,"Deflexión exacta: {res:.2e} m".format(res=abs(flecha))+
                 "\nDeflexión aproximada: {res:.2e} m".format(res=flecha_aprox.numpy()[0])+
                 "\nError relativo: {res:.3f}%".format(res=abs(abs(flecha)-flecha_aprox.numpy()[0])/max(abs(flecha),abs(flecha_aprox.numpy()[0]))*100),
                 wrap = True, fontsize = 8, 
            bbox ={'facecolor':'mediumpurple', 'alpha':.5, 'pad':5})


    plt.xlabel("$[m]$")
    plt.ylabel("$[m]$")
    plt.legend()
    plt.show()



    #PLOT 2: ITERACIONES VS PERDIDA
    
    Adam_its = len(Adam_hist)
    Adam_x_axis = range(Adam_its)
    LBFGS_its = len(LBFGS_hist)
    LBFGS_x_axis = range(Adam_its-1,Adam_its+LBFGS_its-1,1)
      
    
    fig= plt.figure(dpi=220)
    plt.plot(Adam_x_axis,Adam_hist, 'r',linewidth=1, label='Optimizador Adam')
    plt.plot(LBFGS_x_axis,LBFGS_hist, 'b',linewidth=1, label='Optimizador L-BFGS')
        
    plt.title('Historial de pérdidas a lo largo de las iteraciones',fontweight="bold")
    plt.ylabel('valor de Loss')
    plt.xlabel('número de Iteración')
    
    plt.legend()
    plt.show()



    

    #PLOT 3: QQ PLOT
    
    data1 = res
    data2 = -analitica
    
    # Calcula los cuantiles teóricos y observados
    quantiles_data1 = np.percentile(data1, np.linspace(0, 100, 100))
    quantiles_data2 = np.percentile(data2, np.linspace(0, 100, 100))
    
    # Crea el QQ plot
    plt.figure(figsize=(8, 6),dpi = 300)
    plt.scatter(quantiles_data1, quantiles_data2, color='b', alpha=0.6)
    plt.plot([np.min(quantiles_data1), np.max(quantiles_data1)],
             [np.min(quantiles_data1), np.max(quantiles_data1)], color='k', linestyle='--')
    plt.xlabel('respuesta PINN $[m]$')
    plt.ylabel('respuesta analítica $[m]$')
    plt.title('QQ Plot '+ titulo,fontweight="bold")
    
    plt.show()
    









