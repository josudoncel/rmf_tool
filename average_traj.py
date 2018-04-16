import os
import numpy as np
import pandas as pd
from time import time

os.system('make simulate_JSQ')

def averageTraj(N,rho,initial_number_of_jobs,nb_samples):
    fileName = 'traj/averageTraj_N{}_r{}_init{}.npz'.format(
        N,int(rho*100),int(initial_number_of_jobs*10))
    t = time()
    if os.path.exists(fileName):
        loadedFile=np.load(fileName)
        x = loadedFile['x']
        nb_samples_already_computed=loadedFile['nb_samples']
        if nb_samples_already_computed >= nb_samples:
            print('already ',nb_samples_already_computed,'/',
                  nb_samples,' for N=',N,'and rho=',rho)
            return
    else:
        print('no results for N=',N,'and rho=',rho)
        nb_samples_already_computed=0
        x=np.zeros((N*300,20))
    for i in range(nb_samples_already_computed,nb_samples):
        os.system('./simulate_JSQ N{} r{} t{} > traj/tmpFile'.format(
            N,rho,initial_number_of_jobs))
        y=np.array(pd.read_csv('traj/tmpFile',sep=' ',dtype=np.float64))[:,:-1]
        x+=y
        if ((i+1) % 100==0):
            print('\rCompleted:{}/{}, rho={},N={}, estimated remaining time={:.0f}sec'.format(
                i+1,nb_samples,rho,N,
                (nb_samples-i)/(i-nb_samples_already_computed)*(time()-t))
                  ,end='')
    print('\rCompleted:{}/{}, rho={},N={}, total time={:.0f}sec{}'.format(
                nb_samples,nb_samples,rho,N, (time()-t),'                    '))
    # fileName='traj/averageTraj_N{}_r{}_init{}.txt'.format(N,int(rho*100),int(initial_number_of_jobs*10))
    np.savez(fileName,x=x,nb_samples=max(nb_samples,nb_samples_already_computed))


for nb_samples in [100,1000,5000,10000,20000,30000,40000,50000,100000]:
    myN = [10,5,20] if nb_samples<=10000 else [10]
    for N in myN:
        averageTraj(N,0.9,2.8,nb_samples)
        #averageTraj(N,0.7,1.3)      # We cannot see the difference
        averageTraj(N,0.95,3.6,nb_samples)

#x=np.loadtxt('traj/{}'.format(0))
# for i in range(1,100):
#     x+=np.loadtxt('traj/{}'.format(i))
# x /= 100
