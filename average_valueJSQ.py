import os
import numpy as np
import pandas as pd
from time import time

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
os.system('cd {} && make simulate_JSQ'.format(dir_path))

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


def simulateSteadyState(N,rho,d,nb_samples):
    fileName = '{}/steadyState/exp_rho{}_d{}_N{}.npy'.format(dir_path,int(100*rho),d,N)
    if os.path.exists(fileName):
        my_simulations = np.load(fileName)
    else:
        my_simulations = []
    print(len(my_simulations),'already computed')
    if len(my_simulations) >= nb_samples:
        return(my_simulations)
    else:
        my_simulations = list(my_simulations)[0:1050]
        for i in range(len(my_simulations),nb_samples,10):
            tmpFile = '{0}/steadyState/tmpFile'.format(dir_path)
            os.system('{0}/simulate_JSQ r{1} d{2} N{3} e10 > {4}'.format(dir_path,rho,d,N,tmpFile))
            y = list(np.loadtxt(tmpFile))
            print(y)
            for x in y:
                my_simulations.append(x)
        np.save(fileName,np.array(my_simulations))
        return(my_simulations)

def loadAllSteadyStateSimu(N,d,rho):
    fileName='{}/steadyState/exp_rho{}_d{}_N{}.npy'.format(dir_path,int(100*rho),d,N)
    if not os.path.exists(fileName):
        return(simulateSteadyState(N,rho,d,nb_samples=100))
    else:
        return(np.load(fileName))
    
def loadSteadyStateDistributionQueueLength(N,d,rho,returnConfInterval=False):
    myFile = loadSteadyStateDistributionQueueLength(N,d,rho)
    mean = np.mean(myFile,0)
    if not returnConfInterval:
        return(mean)
    else:
        std = np.sqrt(np.var(myFile,0)/len(myFile))
        return(mean,std)
    
def loadSteadyStateAverageQueueLength(N,d,rho,returnConfInterval=False):
    myFile = loadAllSteadyStateSimu(N,d,rho)
    means = np.sum(myFile,1)
    mean=np.mean(means)
    if not returnConfInterval:
        return(mean)
    else:
        std = np.sqrt(np.var(means)/len(means))
        return(mean, std)
    
