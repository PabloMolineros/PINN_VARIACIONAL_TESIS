#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 18:41:40 2021

@author: FMagnani
Editado por Angel Vázquez

Editado por Pablo Molineros y Karen Sacoto

"""

#import sys
#sys.path.insert(0, '../Utils/')

import tensorflow as tf

# funciones que están en Utils en el archivo NeuralNet.py
from NeuralNet import neural_net, PhysicsInformedNN

#%% class neural_net_2out(neural_net):
# Esta clase divide las salidas para luego poder sacar las derivadas de manera sencilla

class neural_net_2out(neural_net):
    
    def __init__(self, wb, lb, layers):
        super(neural_net_2out, self).__init__(wb, lb, layers)

   # call es una función de tf.keras.Sequential (abuela de esta clase)
    # If applicable, update the static input shape of the model.
    def call(self, inputs, training=False):
        """
        Parameters
        ----------
        inputs : Tensor
            Primera columna los xs y segunda columna los ts.
        training : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.
        TYPE
            DESCRIPTION.
        """
        output = super(neural_net_2out, self).call(inputs)
        # https://githwb.com/keras-team/keras/blob/master/keras/engine/sequential.py
        return output[:, 0], output[:, 1]
        # En lugar de devolver la salida como una matriz
        # devuelve la salida de cada neurona de salida.
        # Me parece que esto ayuda a sacar los gradientes con respecto a una
        # sola variable de entrada

#%% VIGA_VARIACIONAL_PINN(PhysicsInformedNN):
class VIGA_VARIACIONAL_PINN(PhysicsInformedNN):

    def __init__(self, x0, w0, x1, x2, w2, x3, w3, wb, lb, layers):
        """

        Parameters
        ----------
        x0 : Tensor que no tiene columnas
            Valores de x 
        w0 : TYPE
            condiciónes iniciales del componente w.

        x_wb : TYPE
            DESCRIPTION.
        x_lb : TYPE
            DESCRIPTION.

        x_f : TYPE
            Puntos de colocación 
            Las filas indican el número puntos de colocación; x_f
            hace referencia a los valores de x de esos puntos de colocación.

        X_star : TYPE
            DESCRIPTION.
        wb : TYPE
            El primer valor es el límite superior espacial y el segundo es el límite superior temporal.
        lb : TYPE
            El primer valor es el límite inferior espacial y el segundo es el límite inferior temporal.
        layers : list
            Contiene valores enteros que corresponden a las neuronas en cada capa.

        Returns
        -------
        None.

        """

        super(VIGA_VARIACIONAL_PINN, self).__init__()
        
        #Características geométricas y mecanicas de la viga
        
        self.q = 60000    #N/m
        self.L = 2.7      #m   
        self.E = 200e9      #Pa 
        self.I = 0.000038929334 # m^4

        # Network architecture
       
        self.model = neural_net(wb, lb, layers)

        # Data initialization
        # Para las condiciónes de contorno
        self.x0 = x0 # x0=0 Es para la condición inicial w(x0=0)=0
        self.w0 = w0 # Este es el valor de w en x0=0. Condición inicial w(x0=0)=0
        self.x_f = x1 # Colocations points
        self.x2 = x2 # Condición de contorno
        self.w2 = w2
        self.x3 = x3 # Condición de contorno
        self.w3 = w3

    
#Definición de la función de perdida       

    def loss(self):
        
        x0 = self.x0
        w0 = self.w0
        w2 = self.w2
        x2 = self.x2
        w3 = self.w3
        x3 = self.x3

        # Loss from supervised learning (at t=0)
        # x0 y t0 son Tensores sin columnas (como vectores)
        # tf.stack le hace un Tensor de dos columnas
        #X0 = tf.stack([x0, t0], axis=1)
        
        
        #Definición de las condiciónes de contorno para la loss
 
        #PROBLEMA 1: Simplemente apoyada
        
        X0 = tf.stack([x0], axis=1) #Condición u(0) = 0        
        w_pred = self.model(X0) #Paso de condición X0 por la red        
        y0 = tf.reduce_mean(tf.square(w0 - w_pred)) #Penalty term de condición X0
        f_w = self.net_f_wv() #Termino de energia para loss         
        yS = tf.reduce_mean(tf.square(f_w)) #Paso de colocation points por la red        
        X2 = tf.stack([x2], axis=1) #Condición u(L) = 0           
        w_pred = self.model(X2) #Paso de condición X2 por la red
        yB = tf.reduce_mean(tf.square(w2-w_pred)) #Penalty term de condición X2

 
        """  
        #PROBLEMA 2: empotrada - libre
        
        X0 = tf.stack([x0], axis=1) #Condición u(0) = 0        
        w_pred = self.model(X0) #Paso de condición X0 por la red        
        y0 = tf.reduce_mean(tf.square(w0 - w_pred)) #Penalty term de condición X0
        f_w = self.net_f_wv() #Termino de energia para loss 
        yS = tf.reduce_mean(tf.square(f_w)) #Paso de colocation points por la red  
        X2 = tf.stack([x2], axis=1) #Condición w_x(0) = 0   
        w_pred, w_x_pred,*_ = self.net_uv(X2) #Paso de condición X2 por la red
        yB = tf.reduce_mean(tf.square(w_x_pred-w2)) #Penalty term de condición X2
        """  
        
        """  

        #PROBLEMA 3: empotrada - simplemente apoyada
        
        X0 = tf.stack([x0], axis=1) #Condición u(0) = 0
        w_pred = self.model(X0) #Paso de condición X0 por la red

        y0 = tf.reduce_mean(tf.square(w0 - w_pred)) #Penalty term de condición X0
        f_w = self.net_f_wv() #Termino de energia para loss 
        yS = tf.reduce_mean(tf.square(f_w)) #Paso de colocation points por la red  
        X2 = tf.stack([x2], axis=1) #Condición w_x(L) = 0   
        w_pred, w_x_pred,*_ = self.net_uv(X2) #Paso de condición X2 por la red
        yB = tf.reduce_mean(tf.square(w_x_pred-w2)) #Penalty term de condición X2
        X3 = tf.stack([x3], axis=1) #Condición u(L) = 0   
        #w_pred, *_ = self.net_uv(X3) #Paso de condición X3 por la red
        w_pred = self.model(X3)
        yC = tf.reduce_mean(tf.square(w_pred-w3)) #Penalty term de condición X3

        """         

        
        #return 5*self.E/self.I*y0    + yS  + 5*self.E/self.I*yB + 5*self.E/self.I*yC #Loss para problema 3
        return 2*self.E/self.I*y0    + yS  + 2*self.E/self.I*yB #Loss para problema 1 y 2

    
    #Calculo de los gradientes para la red neuronal
    
    def net_wv(self, x): #calcula los valores de u

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)

            X = tf.stack([x], axis=1) 
        
            w = self.model(X)
         
            w_x = tape.gradient(w, x)
        
        w_xx = tape.gradient(w_x, x)

        return w, w_x,w_xx


    def net_f_wv(self):
        
        x_f = self.x_f # puntos de colocación

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            

            X_f = tf.stack([x_f], axis=1) 

            w = self.model(X_f)
           
            w_x = tape.gradient(w, x_f)
            
        w_xx = tape.gradient(w_x, x_f)
    
        del tape
    
        #Función de perdida para la ecuación variacional de la viga
        
        f_w = (self.E*self.I/2) * tf.square(w_xx) - self.q * w + 0.5*(self.E*self.I)

    
        return f_w

    # For the final prediction 
    def predict(self, x):
                
        with tf.GradientTape(persistent=True) as tape:    
            tape.watch(x)
            #X = tf.stack([x, t], axis=1) # shape = (N_f,2)
            X = tf.stack([x], axis=1) # shape = (N_f,2)
            
            w = self.model(X)
                
            w_x = tape.gradient(w, X)
            
        w_xx = tape.gradient(w_x, X)
    
        del tape
        
        #Función de perdida para la ecuación variacional de la viga
        
        f_w = (self.E*self.I/2) * tf.square(w_xx) - self.q * w + 0.5*(self.E*self.I)

    
        return w, f_w



