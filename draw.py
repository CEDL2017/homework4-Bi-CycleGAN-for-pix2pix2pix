import numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import pandas as pd

log_file = './checkpoints/monet2photo_cyclegan/loss_log.txt'

epoch = []
DA =[]; GA = []; DB = []; GB = [];
cycA = []; cycB = [];
idtA = []; idtB = [];

f = open(log_file) #sys.argv[1]) #open file
for line in f:
    if "===" not in line:
        infos = line.split(' ')
        epoch.append(infos[1])
        DA.append(infos[7]); GA.append(infos[9]);
        DB.append(infos[13]); GB.append(infos[15]);
        cycA.append(infos[11]); idtA.append(infos[19]);
        cycB.append(infos[17]); idtB.append(infos[21]);
   
################
fig = plt.figure(1)
ax1 = fig.add_subplot(111)#figsize=(15, 10))
ax1.set_title('Gener and Discri Loss of A')#sys.argv[4])
ax1.plot(GA, alpha=0.8, label="G_A")
ax1.plot(DA, 'g', alpha = 0.8, label="D_A")
ax1.set_xlabel('iters')
ax1.set_ylabel('loss')
ax1.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7))


fig = plt.figure(2)
ax1 = fig.add_subplot(111)#figsize=(15, 10))
ax1.set_title('Gener and Discri Loss of B')#sys.argv[4])
ax1.plot(GB, alpha=0.8, label="G_B")
ax1.plot(DB, 'g', alpha = 0.8, label="D_B")
ax1.set_xlabel('iters')
ax1.set_ylabel('loss')
ax1.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7))


fig = plt.figure(3)
ax1 = fig.add_subplot(111)#figsize=(15, 10))
ax1.set_title('Cycle Loss')#sys.argv[4])
ax1.plot(cycA, alpha=0.8, label="cycle_A")
ax1.plot(cycB, 'g', label="cycle_B")
ax1.plot(idtA, 'tab:orange', label="identity_A")
ax1.plot(idtB, 'tab:pink', label="identity_B")
ax1.set_xlabel('iters')
ax1.set_ylabel('loss')
ax1.legend(loc='upper left', bbox_to_anchor=(0.7, 0.7))



plt.show()
