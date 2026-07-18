# Perceptron in python for prototyping

to see further information : read the report

## Goal 
The main goal is to structure our idees, and build the prototype for the project we will later implement in assembly.
The code has to be simple enough to be a proper prototype for the assemby version.


## Structure
The code is divided in three parts : 
- utils/activation_fun.py implements the simple step function
- utils/optimization.py implements the gradient descent as well as the gradient of the perceptron loss function
- main.py implements the training algorithm using a set of known samples as well as the perceptron function that can classify previouly unseen data (and seen since our case is a bit artificial)

