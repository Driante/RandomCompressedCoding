# Random Compressed Coding with Neurons
This repository contains code for reproducing the results from the paper "Random Compressed Coding with Neurons" by Blanco, Malerba et al., 2021. The work investigates neural coding and decoding processes using compressed random encoding strategies in neural networks.

## Overview
The code is organized into different modules, each responsible for different aspects of the model and simulations. Key components include:

Model Structure: The neural network architecture and the encoding/decoding process.
Encoding/Decoding Process: Methods for generating compact neural codes and computing mean squared error (MSE) in different conditions.
3D Encoding: A variant where the first layer encodes a 3D stimulus.
Simulation Scripts: Various simulations are conducted to assess coding performance under different conditions.
Results: The parameters used in each simulation and the results are saved in JLD (Julia Data) files for easy access.
## Repository Structure
src/: Contains the core functions and model definitions for constructing the network and running the encoding/decoding process.

network.jl: Defines the neural network model and includes functions to compute MSE for ideal and neural network-based decoding.
network3D.jl: A variant of the network.jl model, where the first layer encodes a 3D stimulus.
scripts/: Contains different scripts that run simulations under various conditions.

These scripts evaluate coding performance and save the simulation parameters and results in a JLD file.
notebooks/: Jupyter notebooks used for visualizing and plotting results.

These notebooks load the saved JLD files and generate the figures for the analysis.
Requirements
To run the code, you'll need to have the following installed:

Julia: Version 1.x or higher.
JLD: Julia package for saving/loading data in the JLD format. You can install it using import Pkg; Pkg.add("JLD").
PyPlot (for plotting): To visualize results in the Jupyter notebooks.
