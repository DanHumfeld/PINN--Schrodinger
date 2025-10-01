# PINN--Schrodinger
Using co-trained PINNs to solve the Schrodinger problem and the inverse Schrodinger problem.

#
## The Schrodinger Problem, 1 dimension
Given this V(x), solve for Psi<sub>n</sub>(x) and E<sub>n</sub> to satisfy the Schrodinger equation: \[-h<sup>2</sup>/2m Del<sup>2</sup> + V(x)\] Psi<sub>n</sub>(x) = E<sub>n</sub> Psi<sub>n</sub>(x), where h is h<sub>bar</sub>.

+ The problem is solved in atomic units.
+ V(x) is given as the sum of two components. The base component is a soft-walled finite well with a depth of 1000, sigmoid-shaped walls with a steepness of 30, and wall (center) positions of -1.5 and +1.5. The second component is the part added by the user.
+ One NN is used to represent each wavefuntion Psi<sub>n</sub>(x).
+ One NN is used to represent E(n) where n is a continuous variable but E(n) has interpretable meaning only for integer values of n; E(n|n=int) = E<sub>n</sub>.
+ The loss terms used include:
  + Schrodinger equation loss term,
  + wavefunction normalization loss term,
  + wavefunction orthogonality (correlation) loss term,
  + energy difference loss term\*, and
  + energy minimization loss term\*.
+ The orthogonality loss terms are mono-directional; e.g. the orthogonality loss for wavefunctions 1 and 2 is included for training wavefunction 2 but not for training wavefunction 1.
+ \* This code uses Transfer Learning.
  + During Transfer Mode 0, energy minimization is applied and the solutions are forced to have different energy levels. 
  + During Transfer Mode 1, those constraints are both relaxed such that the total loss is based only on the actual requirements of the Schrodinger equation.
+ Code outputs V(x) and each Psi<sub>n</sub>(x), but only outputs E<sub>n</sub> _during training_.
+ Code is _qm_v26.py_.
+ This code file title may imply that the code has anything to do with quantum mechanics, but it doesn't. It's just solving the Schrodinger equation with a numerical approximation. Quantum Mechanics is a set of mathematics used to define and describe solutions - a convenient shorthand that turned into an Algebra, or something else as I forget my formal math definition terms. This code does not deal with quantum mechanics but performs the calculations explicitly that quantum mechanics simplifies.

#
## The Inverse Schrodinger Problem
Given these E<sub>n</sub>, solve for V(x) and Psi<sub>n</sub>(x) to satisfy the Schrodinger equation: \[-h<sup>2</sup>/2m Del<sup>2</sup> + V(x)\] Psi<sub>n</sub>(x) = E<sub>n</sub> Psi<sub>n</sub>(x), where h is h<sub>bar</sub>.

+ The problem is solved in atomic units.
+ V(x) is given as the sum of two components. The base component is a soft-walled finite well with a depth of 1000, sigmoid-shaped walls with a steepness of 30, and wall (center) positions of -1.5 and +1.5. The second component is the part solved for by the code.
+ One NN is used to represent V<sub>2</sub>(x). 
+ One NN is used to represent each wavefuntion Psi<sub>n</sub>(x).
+ One NN is used to represent E(n) where n is a continuous variable but E(n) has interpretable meaning only for integer values of n; E(n|n=int) = E<sub>n</sub>. This NN is pre-trained to the provided E<sub>n</sub>.
+ The loss terms used include:
  + Schrodinger equation loss term,
  + wavefunction normalization loss term,
  + wavefunction orthogonality (correlation) loss term, and
  + energy constraint.
  + Note: energy difference loss term and energy minimization loss term are both defined in the code, but are turned off by the logic in the _Inputs_ section of the code.
+ The orthogonality loss terms are mono-directional; e.g. the orthogonality loss for wavefunctions 1 and 2 is included for training wavefunction 2 but not for training wavefunction 1.
+ The initial wavefunctions are optionally read from models in a specified directory. Current implementation reads the wavefunctions from the soft-walled finite potential well (no V<sub>2</sub>(x)) as the starting wavefunctions.
+ The potential is optionally pre-trained to match a user-provided potential, which is not the potential that will give the correct results but serves to provide an appropriate initial shape to V<sub>2</sub>(x).
+ This code inherited Transfer Learning from _qm_v2.6.py_. It does not use Transfer Learning, i.e. use only Transfer Mode 0. The loss terms are no different between Transfer Modes 0 and 1, due to logic in the _Inputs_ section of the code. The only difference is the file names for the models it loads and saves.
+ Code outputs V(x) and each Psi<sub>n</sub>(x), only outputs E<sub>n</sub> _during training_, but that's okay because these are the problem inputs that are hard-coded in a variable in the _Inputs_ section of the code.
+ Code is _qm_v31.py_.
+ This code file title may imply that the code has anything to do with quantum mechanics, but it doesn't. Same explanation as above.

#
Dependencies:

+ Tensorflow 2.1.0
+ pandas 1.0.1
+ numpy 1.18.1

