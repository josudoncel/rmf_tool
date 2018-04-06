import os
import numpy as np
import pandas as pd

os.system('make simulate_JSQ')

def averageTraj(N,rho,initial_number_of_jobs):
    print('N=',N,'rho=',rho)
    x=np.zeros((N*100,20))
    nb_samples=10000
    for i in range(0,nb_samples):
        os.system('./simulate_JSQ N{} r{} t{} > traj/tmpFile'.format(
            N,rho,initial_number_of_jobs))
        y=np.array(pd.read_csv('traj/tmpFile',sep=' '))[:,:-1]
        x+=y/nb_samples
        if (i % 1000==0): print('{}/{} done rho={}N={}'.format(
                i,nb_samples,rho,N))

    fileName='traj/averageTraj_N{}_r{}_init{}.txt'.format(N,int(rho*100),int(initial_number_of_jobs*10))
    np.savetxt(fileName,x)

for N in [5,10,20]:
    averageTraj(N,0.9,2.7)
    #averageTraj(N,0.7,1.3)

#x=np.loadtxt('traj/{}'.format(0))
# for i in range(1,100):
#     x+=np.loadtxt('traj/{}'.format(i))
# x /= 100
