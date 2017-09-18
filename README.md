**Work in progress**

# Simulation and calcul of Random Systems


This python librarie implements an algorithm to simulate and study a
density dependent population process, to compute its mean-field
approximation and its refined mean-field approximation.

### Remarks




## Documentation

A density dependent process is a Markov chain that evolve in a sub-domain of $R^d$ are where the transition are given by a set $L$ and a list of rate-functions \beta_l. To a system size N is associated a Markov chain whose transitions are (for all $\ell\in L$):

* $x \mapsto x + \frac 1N \ell$ at rate $\beta_\ell(x)$

The class DDPP can be used to defined a process, via the functions *add_transition(l,beta)* and can then be used to produce :

* A sample trajectory (via simulation : function *simulate(N,time)*
* The mean-field (or fluid) approximation : function *ode(time)*
* The refined mean-field approximation (for the steady-state) : function *theoretical_C()*

Most of the functions are documented and their documentation is
accessible by the "help" command.

Apart from that, the documentation is mostly contained in the examples
below (from basic to more advanced). 

### Computation time

The computation of the function 'theoretical_C' grows as d^3, where d is the dimension of the model. It takes about 5 second for a 50-dimensional system, about 40 seconds for a 100-dimensional system.

The simulation of the underlying Markov chain is not optimized and therefore might be slow for large models. 

### Examples

### Small example :
The following code define a $2$-dimensional model and plot the mean-field approximation versus one sample trajectory.

```
import src.rmf_tool as rmf
ddpp = rmf.DDPP()
ddpp.add_transition([-1,1], lambda x : x[0])
ddpp.add_transition([1,-1], lambda x : x[1]+x[0]*x[1])
ddpp.set_initial_state([.5,.5])
ddpp.plot_ODE_vs_simulation(N=100)
```

#### More advanced examples

* [Simple SIR model](BasicExample_SIR.ipynb)
* [2-choice model](Example_2choice.ipynb) 

## Dependencies

This library depends on the following python library:

* numpy
* random
* scipy
* sympy 
* matplotlib.pyplot
