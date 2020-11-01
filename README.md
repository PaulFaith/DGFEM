# DGFEM
DGFEM Solver
El software tiene la siguiente estructura:

main:
  main.py: Este script es el encargado de resolver el problema.
    1-mesh.py: En este modulo se genera la malla.
    2-startup.py: En este modulo se soluciona y se construye la rejilla y la metrica.
    3-advec.py:   En este módulo se soluciona el problema.
  
I'm going to change a little bit the structure of the original algorithm in Matlab to create a Modular structure, then the idea is:

1.-No globals in the algorithm.
2.-Two modules are going to be created, the first module is going to be the Startup module that is going to include all the atributes and methods to create the mesh and the grid, the second module is going to be the solver module, this module is going to solve the advection problem for an specific time and also is going to create a group of 4 plots for 4 different solution times.

the class startup is going to need just the basic data to create the mesh and all the extra constants and variables are goint to be declared inside the class.
the class sovler is going to be instantiated passing to it the startup instantiated class and the initial conditions, this will help to have all the atributes ready to use inside the solver object to calculate the different solutions for the different times and plot it.
