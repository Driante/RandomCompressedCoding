  Code for reproducing the results of "Compact Neural Codes".
  The code is structured in the following way. 
  The source code contains the structure for constructing the model and the encoding decoding process.
  network.jl contains the definition of a network and functions for computing the MSE in an ideal and  with a NN.
  network3D.jl  the same, but the first layer encode now a 3D stimulus
  
  Different scripts run different simulations and compute the coding performances according to various conditions. They save the parameters used in the simulation and the results in a JLD file in a local repository.
  Jupyter notebooks are used to plot the figures and save them.
