import os
import numpy as np
import pandas as pd
from time import time

os.system('cd {} && make simulate_JSQ'.format(dir_path))
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

def simulateAverageTraj(N,rho,initial_number_of_jobs,nb_samples):
    fileName = '{}/traj/averageTraj_N{}_r{}_init{}.npz'.format(
        dir_path,N,int(rho*100),int(initial_number_of_jobs*10))
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
        os.system('{0}/simulate_JSQ N{1} r{2} t{3} > {0}/traj/tmpFile'.format(
            dir_path,N,rho,initial_number_of_jobs))
        y=np.array(pd.read_csv('{}/traj/tmpFile'.format(dir_path),sep=' ',dtype=np.float64))[:,:-1]
        x+=y
        if ((i+1) % 100==0):
            print('\rCompleted:{}/{}, rho={},N={}, estimated remaining time={:.0f}sec'.format(
                i+1,nb_samples,rho,N,
                (nb_samples-i)/(i-nb_samples_already_computed)*(time()-t))
                  ,end='')
    print('\rCompleted:{}/{}, rho={},N={}, total time={:.0f}sec{}'.format(
                nb_samples,nb_samples,rho,N, (time()-t),'                    '))
    np.savez(fileName,x=x,nb_samples=max(nb_samples,nb_samples_already_computed))


def loadTransientSimu(N,rho,initial_number_of_jobs,nb_samples=100):
    fileName='{}/traj/averageTraj_N{}_r{}_init{}.npz'.format(
        dir_path,N,int(rho*100),int(10*initial_number_of_jobs))
    if not os.path.exists(fileName) :
        print('No data file found : we need to simulate')
        simulateAverageTraj(N,rho,initial_number_of_jobs,nb_samples)
    myData = np.load(fileName)
    if myData['nb_samples'] < nb_samples:
        simulateAverageTraj(N,rho,initial_number_of_jobs,nb_samples)
        myData = np.load(fileName)
    
    Y = myData['x']/myData['nb_samples']
    Tsimu = np.arange(0,300*N)/(N*(1+rho))
    print('average over ',myData['nb_samples'],'simulations')
    return(Tsimu,Y)
