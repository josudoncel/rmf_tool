import numpy as np
import random as rnd
import scipy.integrate as integrate
import sympy as sym
import scipy.linalg 
import numpy.linalg 


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
        if self._x0 == None:
            raise InitialConditionNotDefined
        def drift(x):
            return (sum([self._list_of_transitions[i]*self._list_of_rate_functions[i](x) for i in range(len(self._list_of_transitions))],0))
        
        T = np.linspace(0,time,number_of_steps)
        X = integrate.odeint( lambda x,t : drift(x), self._x0, T)
        return(T,X)

        
    def fixed_point(self):
        """Computes the fixed of the ODE (if this ODE has a fixed point starting from x0)
        """
        return(self.ode(1000)[1][-1,:])

    def theoretical_C(self):
        n = self._model_dimension
        number_transitions = len(self._list_of_transitions)
        Xstar = self.fixed_point()
    
        Var=np.array([sym.symbols('x_{}'.format(i)) for i in range(n)])
    
        f_x=np.zeros(n)
        for i in range(number_transitions):
            f_x = f_x + self._list_of_transitions[i]*self._list_of_rate_functions[i](Var)
        
        print('SOMETHING TO BE DONE HERE')
        dim = n-1
        for i in range(n):
            f_x[i]=f_x[i].subs(Var[-1],1-sum(array([Var[i] for i in range(n-1)])))
        
        A=array([[sym.lambdify(Var ,sym.diff(f_x[i],Var[j]))(*[Xstar[k]
                                                               for k in range(n)]) 
                  for j in range(dim)]
                 for i in range(dim)])

        B=array([[[sym.lambdify(Var,sym.diff(f_x[j],Var[k],Var[l]))(*[Xstar[i] 
                                                                      for i in range(n)]) 
                   for l in range(dim)] 
                  for k in range(dim)] 
                 for j in range(dim)])
        
        Q=array([[0. for i in range(dim)] for j in range(dim)])

        for l in range(number_transitions):
            Q += array([[self._list_of_transitions[l][p]*self._list_of_transitions[l][m]*
                         self._list_of_rate_functions[l](Xstar)
                         for m in range(dim)]
                    for p in range(dim)])

        W = scipy.linalg.solve_lyapunov(A,Q)
        A_inv=numpy.linalg.inv(A)
    
        C=[ 0.5*sum([A_inv[i][j]*sum(array([[B[j][k_1][k_2]*W[k_1][k_2] 
                                             for k_2 in range(dim)] 
                                            for k_1 in range(dim)])) 
                     for j in range(dim)]) 
            for i in range(dim)]
        return(np.sum(C,1))
    
    
    def compare_ODE_simulation():
        raise NotImplemented
