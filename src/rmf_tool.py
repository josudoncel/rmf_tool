import numpy as np
import random as rnd
import scipy.integrate as integrate
import sympy as sym
import scipy.linalg 
import numpy.linalg
import matplotlib.pyplot as plt


class RmfError(Exception):
    """Basic error class for this module
    """
class DimensionError(RmfError):
    pass
class NotImplemented(RmfError):
    pass
class NegativeRate(RmfError):
    pass
class InitialConditionNotDefined(RmfError):
    pass

class DDPP():
    """ DDPP serves to define and study density depend population processes
    
    """

    def __init__(self):
        """
        Explain params 
        
        Args:
            param1 (str): Description of `param1`.
            param2 (:obj:`int`, optional): Description of `param2`. Multiple
                lines are supported.
        """
        self._list_of_transitions = []
        self._list_of_rate_functions = []
        self._x0 = None
        self._model_dimension = None

    def add_transition(self,l,f):
        """
        Add a new transition of the form (\ell,\beta_\ell), where \beta_\ell(x) is the rate of the transition

        Args:
            l: A vector of changes (of size n)
            f: A function from R^n \to R, where f(x) is the rate at which the transition occurs

        Returns:
            True if successful, False otherwise 

        Raises: 
            DimensionError if l is not of the right size.
        """
        if self._model_dimension is not None and self._model_dimension != len(l):
            raise DimensionError
        self._model_dimension = len(l)
        self._list_of_transitions.append(np.array(l))
        self._list_of_rate_functions.append(f)

    def set_initial_state(self,x0):
        if self._model_dimension and self._model_dimension != len(x0):
            raise DimensionError
        self._model_dimension = len(x0)
        self._x0 = x0

    def simulate(self,N,time):
        """
        Simulates an realization of the stochastic process with N objects 

        Returns:
            (T,X), where : 
            - T is a 1-dimensional numpy array, where T[i] is the time of the i-th time step. 
            - x is a 2-dimensional numpy array, where x[i,j] is the j-th coordinate of the system at time T[i]
        """
        if N=='inf':
            return(ode(time))
        if self._x0 is None :
            raise InitialConditionNotDefined 
        nb_trans=len(self._list_of_transitions)
        t=0
    
        #if fix!=-1:     seed(fix)
        x = np.array(self._x0)
        T = [0]
        X = [x]
        while t<time:
            L_rates=[self._list_of_rate_functions[i](x) for i in range(nb_trans)]
            if any(rate<-1e-14 for rate in L_rates):
               raise NegativeRate
            S=sum(L_rates)
            if S<=1e-14:
                print('System stalled (total rate = 0)')
                t = time
            else:
                a=rnd.random()*S
                l=0
                while a > L_rates[l]:
                    a -= L_rates[l]
                    l += 1
        
                x = x+(1./N)*self._list_of_transitions[l]

                t+=rnd.expovariate(N*S)
            
            T.append(t)
            X.append(x)
    
        X = np.array(X)
        return(T,X)
    
    def ode(self,time,number_of_steps=1000):
        """Simulates the ODE (mean-field approximation)
        """
        if self._x0 is None:
            raise InitialConditionNotDefined
        def drift(x):
            return (sum([self._list_of_transitions[i]*self._list_of_rate_functions[i](x) for i in range(len(self._list_of_transitions))],0))
        
        T = np.linspace(0,time,number_of_steps)
        X = integrate.odeint( lambda x,t : drift(x), self._x0, T)
        return(T,X)

        
    def fixed_point(self):
        """Computes the fixed of the ODE (if this ODE has a fixed point starting from x0)
        """
        if self._x0 is None:
            print('No initial condition given. We assume that the initial condition is "x0=[1,0,...]"')
            self._x0 = np.zeros(self._model_dimension)
            self._x0[0] = 1
        return(self.ode(time=10000,number_of_steps=10)[1][-1,:])

    def doTransitionsConserveSum(self):
        """This function tests if the transitions conserve the sum of the coordinates.

        Returns : True or False.
        """
        for l in self._list_of_transitions:
            if sum(l) != 0:
                return False
        return(True)

    def test_for_linear_dependencies(self):
        """This function tests if there the transition conserve some subset of the space. 
        
        Return (dim, subs) where
        * dim is the rank of the matrix of transitions
        * subs corresponds to the null space of the transitions"""
        
        M = np.array([l for l in self._list_of_transitions])
        u,d,v = np.linalg.svd(M,full_matrices=False)
        dim = sum(abs(d) > 1e-8)
        if dim < self._model_dimension:
            v = v[dim:]
            p,l,u = scipy.linalg.lu(v)
            subs=u
            for i in range(subs.shape[0]):
                j = np.argmax( abs(subs[i,:])>1e-5)
                subs[i,:] = subs[i,:] / subs[i,j]
        else:
            subs = np.array([])
        return(dim,subs)

    def theoretical_C(self):
        """ To be written 
        """
        n = self._model_dimension
        number_transitions = len(self._list_of_transitions)
        Xstar = self.fixed_point()
    
        Var=np.array([sym.symbols('x_{}'.format(i)) for i in range(n)])
        f_x=np.zeros(n)
        for i in range(number_transitions):
            f_x = f_x + self._list_of_transitions[i]*self._list_of_rate_functions[i](Var)


        # The following code attempts to reduce the number of dimensions by testing
        # if the transitions conserve some linear combinations of the coordinates
        (dim,subs) = self.test_for_linear_dependencies()
        variables = [i for i in range(n)]
        for sub in range(subs.shape[0]):
            j = np.argmax( abs(subs[sub,:])>1e-5)
            variables.remove(j)
            for i in range(n):
                f_x[i]=f_x[i].subs(Var[j],sum([self._x0[k]*subs[sub,k] for k in range(j,n)])
                                   -sum(np.array([Var[k]*subs[sub,k] for k in range(j+1,n)])))
        Var = [Var[i] for i in variables]
        
        dF = [[sym.diff(f_x[variables[i]],Var[j]) for j in range(dim)] for i in range(dim)]
        subs_dictionary = {Var[i]:Xstar[variables[i]] for i in range(dim)}
        A=np.array([[float(dF[i][j].evalf(subs=subs_dictionary))
                  for j in range(dim)]
                 for i in range(dim)])
        B=np.array([[[float(sym.diff(dF[j][k],Var[l]).evalf(subs=subs_dictionary) ) 
                      for l in range(dim)]
                     for k in range(dim)]
                    for j in range(dim)])
        Q=np.array([[0. for i in range(dim)] for j in range(dim)])

        for l in range(number_transitions):
            Q += np.array([[self._list_of_transitions[l][variables[p]]*self._list_of_transitions[l][variables[m]]*
                         self._list_of_rate_functions[l](Xstar)
                         for m in range(dim)]
                    for p in range(dim)])
        
        W = scipy.linalg.solve_continuous_lyapunov(A,Q)
        A_inv=numpy.linalg.inv(A)
        BtimesW = [sum(np.array([[B[j][k_1][k_2]*W[k_1][k_2] 
                           for k_2 in range(dim)] 
                          for k_1 in range(dim)])) for j in range(dim)]
                
        C=[ 0.5*sum([A_inv[i][j]* BtimesW[j] for j in range(dim)]) 
            for i in range(dim)]
        C = np.sum(C,1)

        # We now attemps to reconstruct the full C if the number of dimensions was reduced. 
        if dim < n :
            newC = np.zeros(n)
            for i in range(dim):
                newC[variables[i]] = C[i]
            for sub in reversed(range(len(subs))):
                j = np.argmax( abs(subs[sub,:])>1e-5)
                newC[j] = -sum(np.array([newC[k]*subs[sub,k] for k in range(j+1,n)]))
            C = newC
        return(np.array(C))


    def _batch_meanConfidenceInterval(self,T,X):
        n = len(T)
        if len(T)<300:  # We do not do batch if we have less than 200 samples
            return(np.sum(np.diff(T[int(n/2):n])*X[int(n/2):n-1]) / (T[n-1]-T[int(n/2)]),None)
        # Otherwise we split the data into 100 batchs of size 
        Y = [np.sum(np.diff(T[int(2*i*n/300):int((2*i+1)*n/300)])*
                    X[int(2*i*n/300):int((2*i+1)*n/300)-1] /
                    (T[int((2*i+1)*n/300)]-T[int(2*i*n/300)])) for i in range(50,150)]
        return (np.mean(Y),2*np.std(Y)/10)
    
                                      
    def steady_state_simulation(self,N,time=1000):
        """Generates a sample of E[X] in steady-state.

        The expectation is computed by performing one simulation from
        t=0 to t=time and averaging the values over time/2 .. time
        (the implicit assumption is that the system should be roughly
        in steady-state at time/2).

        """
        T,X = self.simulate(N,time)
        n = len(T)
        n2 = int(n/2)
        result = np.array([self._batch_meanConfidenceInterval(T,X[:,i]) for i in range(len(X[0,:]))])
        return(result[:,0],result[:,1])
        
    def compare_refinedMF(self,N,time=1000):
        """Compare E[X] with its mean-field and refined mean-field approximation  

        Args : 
             N (int) : system's size
             time : Computes the expectation by

        Return : (Xm,Xrmf,Xs,Vs) where: 
             Xm is the fixed point of the mean-field approximation 
             Xrmf = Xm + C/N
             Xs is an approximation E[X] (computed by simulation)
             Vs is an estimation of the variance
        
        """
        Xm = self.fixed_point()
        C = self.theoretical_C()
        Xrmf = Xm+C/N
        Xs,Vs = self.steady_state_simulation(N,time)
        return(Xm,Xrmf,Xs,Vs)
    
    def plot_ODE_vs_simulation(self,N,time=10):
        """Plot the ODE and the simulation on the same graph
        
        The system is simulated from t=0 to t=time (both for the
        stochastic and ODE version), starting from x0. 
        
        Args: 
             N (int) : system's size
             time : the time until the simulation should be run

        """
        T,X = self.simulate(N,time=10)
        plt.plot(T,X)
        plt.gca().set_prop_cycle(None)
        T,X = self.ode(time=10)
        plt.plot(T,X,'--')
        plt.legend(['x_{}'.format(i) for i in range(self._model_dimension)])
        plt.xlabel('Time')
        plt.ylabel('x_{i}')
        plt.show()
