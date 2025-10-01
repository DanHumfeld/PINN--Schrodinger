#################################################################
# Code      Quantum Mechanics via PINN with FI
# Version   2.6
# Author    Dan Humfeld, DanHumfeld@Yahoo.com
# Note      This version is meant to solve the problem "find the 
#           solution to the 1D Schrodinger equation given this 
#           V(x)" which will be much like a finite well.
#
#################################################################
# Inputs 
#################################################################   
# Training Mode: Train new = 0, continue training = 1, only load models = anything greater
train_mode = 0
# Transfer Mode: None defined
transfer_mode = 0

# Path for model files
path_for_models = 'Working Models v2.6c/'

# Epoch and batch size control
epochs = 200000
batch_max = 1000
batch_min = 20
loss_save_threshold = 5.0e-3

# Loss-related parameters
initial_energy_target = 1.0
energy_target_factor = 0.9998
normalizing = True
override_transfer_mode = False
using_normalization_loss = True
minimizing_energy = True
requiring_different_energy_levels = True
minimum_energy_level_ratio = 1.10
if (not override_transfer_mode):
    if (transfer_mode == 0):
        minimizing_energy = True
        requiring_different_energy_levels = True
    if (transfer_mode == 1):
        minimizing_energy = False
        requiring_different_energy_levels = False

# Model hyper-parameters
nodes_per_layer = 32
model_layers = 3
wavefunction_count = 7

# Operating options
time_reporting = True
weighted_loss_reporting = True

#################################################################
# Importing Libraries
#################################################################

# # Enable this section to hide all warnings and errors
import os
import logging
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import sys
stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import absl.logging

import math
import random
import numpy as np
from numpy import savetxt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Multiply
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from time import time
start_time = time()

#################################################################
# More Inputs
#################################################################
# Optimizer for training: SGD, RSMProp, Adagrad, Adam, Adamax...
learning_rate = 0.001 #0.0005
my_optimizer = optimizers.Adam(learning_rate)
initializer = 'glorot_uniform'

#################################################################
# Problem
#################################################################
x_min = -2.
x_max = 2.
x_wall_1 = -1.5
x_wall_2 = 1.5
steepness = 30
depth = 1000
def v_definition(x):
    v = depth * (tf.sigmoid(steepness*(x_wall_1 - x)) + tf.sigmoid(steepness*(x - x_wall_2)))
    #v2 = 0.
    v2 = depth * 0.25 - depth * 0.25 * (tf.sigmoid(steepness*(0.35 - x)) + tf.sigmoid(steepness*(x - 0.45)))
    v3 = 0.
    #v2 = tf.sin(2*math.pi*x) * 30.0
    #v3 = 8.0 * x
    #print([type(v), type(v2), type(v3)])
    #v = tf.add(v, tf.add(v2, v3))
    v = tf.add(v, v2)
    v = tf.add(v, v3)
    return v

#################################################################
# Constants
#################################################################
hbar = 1
m = 1

#################################################################
# Constraints
#################################################################

#################################################################
# Normalize Constraints
#################################################################

#################################################################
# File names
#################################################################
if (train_mode == train_mode):
    output_model_file_names = []
    output_energy_file_name = path_for_models + 'mode_' + str(transfer_mode) + '_e.keras'
    for i in range(wavefunction_count):
        output_model_file_names.append(path_for_models + 'mode_' + str(transfer_mode) + '_psi' + str(i) + '.keras')
    prediction_results = path_for_models + 'mode_' + str(transfer_mode) + '_solution.csv'
    prediction_results_file_names = []
    prediction_results_file_names.append(prediction_results)
    loss_history = path_for_models + 'mode_' + str(transfer_mode) + '_loss_history.csv'
    input_model_file_names = []

if ((train_mode == 0) and (transfer_mode > 0)):
    load_transfer_mode = transfer_mode - 1
else:
    load_transfer_mode = transfer_mode
input_energy_file_name = path_for_models + 'mode_' + str(load_transfer_mode) + '_e.keras'
for i in range(wavefunction_count):
    input_model_file_names.append(path_for_models + 'mode_' + str(load_transfer_mode) + '_psi' + str(i) + '.keras')


#################################################################
# Building or Load Models
#################################################################
def new_model(dimensions, layers, nodes, primary_activation, final_activation):
    model = Sequential()
    model.add(Dense(nodes, input_dim = dimensions, activation = primary_activation, kernel_initializer = initializer, bias_initializer = initializer))
    for layer_number in range(layers-1):
        model.add(Dense(nodes, activation = primary_activation, kernel_initializer = initializer, bias_initializer = initializer))
    model.add(Dense(1, activation = final_activation, kernel_initializer = initializer, bias_initializer = initializer))
    model.compile(loss = 'mse', optimizer = my_optimizer)
    return model

def load_models(models, file_names):
    for i in range(len(models)):
        models[i] = keras.models.load_model(file_names[i])
    return

def save_models(models, file_names):
    for i in range(len(models)):
        models[i].save(file_names[i])
    return

models = []
for model_number in range(wavefunction_count):                                        # Initialize new models, even if they will be overwritten via loading
    models.append(new_model(1, model_layers, nodes_per_layer, 'tanh', 'linear'))
energy = new_model(1, model_layers, nodes_per_layer, 'relu', 'linear')                # Energy(n), not Energy(x)
if ((transfer_mode > 0) or (train_mode >= 1)):
    load_models(models, input_model_file_names)
    energy = keras.models.load_model(input_energy_file_name)
save_models(models, output_model_file_names)
save_models([energy], [output_energy_file_name])

#################################################################
# Main Code
#################################################################
if (train_mode in [0, 1]):
    print("Training mode = ", train_mode)
    print("Transfer mode = ", transfer_mode)

    training_process_data = {'epochs': [], 'loss': [], 'time': []}
    min_loss = 100
    batch = batch_max
    last_time = time()
    energy_target = initial_energy_target

    for current_epoch in range(0, epochs):
        # Create tensors to feed to TF 
        x_arr = np.random.uniform(x_min, x_max, batch)
        x_arr = np.sort(x_arr)
        x_feed = np.column_stack((x_arr)) 
        x_feed = tf.Variable(x_feed.reshape(len(x_feed[0]),1), trainable=True, dtype=tf.float32)

        n_feed = np.column_stack(np.arange(wavefunction_count))
        n_feed = tf.Variable(n_feed.reshape(len(n_feed[0]),1), trainable=True, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape_3:    
            with tf.GradientTape(persistent=True) as tape_2:  
                with tf.GradientTape(persistent=True) as tape_1:
                    # Watch parameters
                    tape_1.watch(x_feed)
                    tape_1.watch(n_feed)

                    # Define functions
                    psi = [model([x_feed]) for model in models]
                    e = energy([n_feed])

                # Watch parameters
                tape_2.watch(x_feed)    
                tape_2.watch(n_feed)                
                
                # Take derivitives
                dpsi_dx = [tape_1.gradient(wavefunction, x_feed) for wavefunction in psi]

            # Watch parameters
            tape_3.watch(x_feed)
            tape_3.watch(n_feed)

            # Take derivative
            d2psi_dx2 = [tape_2.gradient(dpsi_dxn, x_feed) for dpsi_dxn in dpsi_dx]

            # Normalize, approximately
            x_roll = tf.roll(x_feed,1,0)
            x_width = k.get_value(x_feed - x_roll)
            x_width[0] = 0 
            norms = [[k.sum(tf.multiply(x_width, tf.multiply(models[m]([x_feed]), models[n]([x_feed])))) for n in range(len(models))] for m in range(len(models))]
            if normalizing:
                psi = [tf.divide(psi[num], tf.sqrt(norms[num][num])) for num in range(len(psi))]
                d2psi_dx2 = [tf.divide(d2psi_dx2[num], tf.sqrt(norms[num][num])) for num in range(len(d2psi_dx2))]

            # Calculate Schrodinger loss
            v = v_definition(x_feed)
            loss_Schrodinger_list = tf.multiply(-((hbar**2)/2/m),d2psi_dx2) + tf.multiply(v, psi) - tf.multiply(e[:,None,:], psi)
            loss_Schrodinger_list = k.square(loss_Schrodinger_list)
            loss_Schrodinger_wavefunction = [k.mean(loss_Schrodinger_list[n]) for n in range(wavefunction_count)]

            # Calculate normalization loss
            loss_normalization_wavefunction = [k.square(1.0 - norms[n][n]) for n in range(len(norms))]

            # Enact loss to require correlations to be zero, with directionality so psi0 doesn't care about psi1 but psi1 cares about psi0
            loss_correlation_2d = [[tf.constant(0, dtype=tf.float32) if (n >= m) else tf.divide(tf.divide(norms[n][m], tf.sqrt(norms[n][n])),tf.sqrt(norms[m][m])) for n in range(len(norms))] for m in range(len(norms))]
            loss_correlation_wavefunction = [k.mean(k.square([loss_correlation_2d[n][m] for m in range(len(norms))])) for n in range(len(norms))]
            # Note that this list does not get averaged; these are different loss terms for each of the different wavefunctions, as required for directionality

            # Calculate loss term to require higher energy than in provided list
            #loss_energy_difference_wavefunction = [tf.constant(0, dtype=tf.float32) if (n == 0) else tf.reshape(k.square(k.elu(tf.subtract(minimum_energy_level_ratio * e[n-1], e[n]))), []) for n in range(wavefunction_count)]
            if (requiring_different_energy_levels):
                loss_energy_difference_wavefunction = [tf.constant(0, dtype=tf.float32) if (n == 0) else tf.reshape(k.square(k.relu(tf.subtract(minimum_energy_level_ratio * e[n-1], e[n]))), []) for n in range(wavefunction_count)]
            else:
               loss_energy_difference_wavefunction = [tf.constant(0, dtype=tf.float32) for n in range(wavefunction_count)]

            # Calculate loss term to minimize energy
            if (current_epoch == 0):
                energy_target = [k.get_value(e[n]) for n in range(wavefunction_count)]
            loss_energy_minimization_wavefunction = [tf.reshape(k.square(energy_target[n] - e[n][0]), []) for n in range(wavefunction_count)]
            energy_target = [e[n] * energy_target_factor for n in range(len(energy_target))]

            # Weight losses
            loss_names = ["Schrodinger", "normalization", "correlations", "energy_difference", "energy_minimization"]
            #losses = [loss_Schrodinger, loss_normalization, loss_correlation_list, loss_energy_difference_list, loss_energy_minimization]
            losses_wavefunction = [[loss_Schrodinger_wavefunction[n], loss_normalization_wavefunction[n], loss_correlation_wavefunction[n], loss_energy_difference_wavefunction[n], loss_energy_minimization_wavefunction[n]] for n in range(wavefunction_count)]
            loss_weightings = np.ones(len(losses_wavefunction[0]))
            #loss_weightings[0] = 1
            #loss_weightings[1] = 1.0e1
            loss_weightings[2] = 1.0e2
            loss_weightings[3] = 1.0e2
            if (not using_normalization_loss):
                loss_weightings[1] = 0
            if (not requiring_different_energy_levels):
                loss_weightings[3] = 0
            if (not minimizing_energy):
                loss_weightings[4] = 0
            #loss_total = sum([x*y for (x,y) in zip(losses, loss_weightings)])
            #print(losses_wavefunction[0])
            #print([x, y] for (x,y) in zip(losses_wavefunction[0], loss_weightings))
            #loss_total_0 = sum([x*y for (x,y) in zip(losses_wavefunction[0], loss_weightings)])
            #print(loss_total_0)
            loss_total_wavefunction = [sum([x*y for (x,y) in zip(losses_wavefunction[n], loss_weightings)]) for n in range(wavefunction_count)]
            losses_weighted_wavefunction = [[x*y for (x,y) in zip(losses_wavefunction[n], loss_weightings)] for n in range(wavefunction_count)]
            losses_weighted = [sum(losses_weighted_wavefunction[n]) for n in range(wavefunction_count)]
            loss_total = sum(loss_total_wavefunction)

            #print(["{:.3e}".format(k.get_value(loss)) for loss in losses_weighted])
            
        # Train the model
        #gradients = [tape_3.gradient(loss_total, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for model in models]
        gradients = [tape_3.gradient(loss_total_wavefunction[n], models[n].trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for n in range(wavefunction_count)]
        for model_num in range(len(models)):
            my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients[model_num], models[model_num].trainable_variables) if grad is not None) 
        #gradients_e = [tape_3.gradient(loss_total, model.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO) for model in energies]
        gradients_e = tape_3.gradient(loss_total, energy.trainable_variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        my_optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients_e, energy.trainable_variables) if grad is not None) 

        # Take a break and report
        #i_loss = k.get_value(sum(losses_weighted))[0]
        current_loss = k.get_value(loss_total)
        #if i % 1 == 0:
        #if i % 10 == 0:
        if current_epoch % 100 == 0:
            print("Step " + str(current_epoch) + " -------------------------------")
            print("[loss_Schrodinger, loss_normalization, loss_correlation_list, loss_energy_difference_list, loss_energy_minimization]")
            for model_num in range(wavefunction_count):
                #print(losses_wavefunction[model_num])
                #print("Wavefunction", model_num, ["{:.3e}".format(k.get_value(loss)) for loss in losses_wavefunction[model_num]])
                print("Wavefunction", model_num, ["{:.3e}".format(k.get_value(loss)) for loss in losses_weighted_wavefunction[model_num]])
            print("Energies", ["{:.3e}".format(k.get_value(e[n][0])) for n in range(wavefunction_count)])
            print("Loss_tot: ", "{:.3e}".format(current_loss))
            if (time_reporting):
                print("Calculation time for last period: ", "{:.0f}".format(round(time() - last_time, 0)), "    Batch size: ", "{:.0f}".format(batch))
            last_time = time()
            
            # Save losses
            current_time = round(time() - start_time, 2)
            training_process_data['epochs'].append(current_epoch)
            training_process_data['loss'].append(current_loss)
            training_process_data['time'].append(current_time)
            if weighted_loss_reporting:
                data_to_write = np.array([current_epoch, current_loss])
                header_text = "Epoch,Total Loss"
                #data_to_write = np.concatenate((data_to_write, [loss_value for n in range(wavefunction_count) for loss_value in losses_wavefunction[n]]))
                data_to_write = np.concatenate((data_to_write, [loss_value for n in range(wavefunction_count) for loss_value in losses_weighted_wavefunction[n]]))
                weighted_loss_names = [name+str(n) for n in range(wavefunction_count) for name in loss_names]
                #print(weighted_loss_names)
                header_text = header_text + ',' + ','.join(weighted_loss_names)
                if (current_epoch == 0):
                    np.savetxt(loss_history, data_to_write.reshape(-1,1).T, header = header_text, delimiter = ',', fmt='%f')
                else:
                    with open(loss_history, 'a') as file:
                        np.savetxt(file, data_to_write.reshape(-1,1).T, fmt = '%f', delimiter=',')
            else:
                np.savetxt(loss_history, np.column_stack((training_process_data['epochs'], training_process_data['loss'])),
                        comments="", header="Epoch,Total Loss", delimiter=',', fmt='%f')

            # Only save model if loss is improved
            if (min_loss > current_loss):
                min_loss = current_loss
            if (current_loss is not math.nan):
                if ((current_loss < loss_save_threshold) or (min_loss > loss_save_threshold)):
                    save_models(models, output_model_file_names)
                    save_models([energy], [output_energy_file_name])
                    # Add content here to write the prediction every time it saves
                    nodes = batch_max
                    s_feed = x_min + (x_max - x_min) * (np.arange(nodes)/(nodes-1))
                    report_time = []
                    results = s_feed
                    results = np.column_stack((results, v_definition(s_feed)))
                    for model_num in range(len(models)):
                        next_output = np.reshape(tf.divide(models[model_num].predict([s_feed]), tf.sqrt(norms[model_num][model_num])),(nodes))
                        results = np.column_stack((results, next_output))
                    np.savetxt(prediction_results, results, delimiter=',') 
            
            # Adjust batch
            #batch = min(batch_max, max(batch_min, math.ceil(5 * i_loss**-1)))

    save_models(models, output_model_file_names)
    save_models([energy], [output_energy_file_name])

#################################################################
# Predicting
#################################################################
# Inputs
nodes = batch_max
x_feed = x_min + (x_max - x_min) * (np.arange(nodes)/(nodes-1))
report_time = []

results = x_feed
results = np.column_stack((results, v_definition(x_feed)))

# Determine normalization
x_roll = tf.roll(x_feed,1,0)
x_width = k.get_value(x_feed - x_roll)
x_width[0] = 0 
norms = [[k.sum(tf.multiply(x_width, tf.multiply(models[m].predict([x_feed]), models[n].predict([x_feed])))) for n in range(len(models))] for m in range(len(models))]

for model_num in range(len(models)):
    next_output = np.reshape(tf.divide(models[model_num].predict([x_feed]), tf.sqrt(norms[model_num][model_num])),(nodes))
    results = np.column_stack((results, next_output))

np.savetxt(prediction_results, results, delimiter=',') 
# Doesn't currently output predicted energies

print("Job's done")
