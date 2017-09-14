import numpy as np
import random as rnd
import scipy.integrate as integrate
# from numpy import array
# from random import random, expovariate
# from scipy.integrate import odeint

class RmfError(Exception):
    """Basic error class for this module
    """
class DimensionError(RmfError):
    pass
class NotImplemented(RmfError):
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
        if self._model_dimension != None and self._model_dimension != len(l):
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
        if self._x0 == None :
            raise InitialConditionNotDefined 
        nb_trans=len(self._list_of_transitions)
        t=0
    
        #if fix!=-1:     seed(fix)
        x = np.array(self._x0)
        T = [0]
        X = [x]
        while t<time:
            L_poids=[self._list_of_rate_functions[i](x) for i in range(nb_trans)]
            S=sum(L_poids)
            if S<=1e-14:
                print('System stalled (total rate = 0)')
                t = time
            else:
                a=rnd.random()*S
                l=0
                while a > L_poids[l]:
                    a -= L_poids[l]
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
        def drift(x):
            return (sum([self._list_of_transitions[i]*self._list_of_rate_functions[i](x) for i in range(len(self._list_of_transitions))],0))
        
        T = np.linspace(0,time,number_of_steps)
        X = integrate.odeint( lambda x,t : drift(x), self._x0, T)
        return(T,X)

        
    def fixed_point(self,x0=None):
        """Computes the fixed of the ODE (if this ODE has a fixed point starting from x0)
        """
        raise NotImplemented

    def theoretical_C(self):
        raise NotImplemented

    def compare_ODE_simulation():
        raise NotImplemented
