# ICANN2021 - Separation of Memory and Processing in Dual Recurrent Neural Networks

## Authors:
Christian Oliva - christian.oliva@estudiante.uam.es

Luis F. Lago-Fernández - luis.lago@uam.es

*Escuela Politécnica Superior, Universidad Autónoma de Madrid. 20849 Madrid, Spain*

## Description:
This repository contains the code and the results of the experiments described in our paper: the results for the complete set of *Tomita Grammars* and the DualRNN code in **Tensorflow 1.7**. We also have included two notebooks for an easier visualization of our results.

## Abstract:
We explore a neural network architecture that stacks a recurrent layer and a feedforward layer, both connected to the input. We compare it to a standard recurrent neural network. When noise is introduced into the recurrent units activation function, the two networks display binary activation patterns that can be mapped into the discrete states of a finite state machine. But, while the former is equivalent to a Moore machine, the latter can be interpreted as a Mealy machine. The additional feedforward layer reduces the computational load on the recurrent layer, which is used to model the temporal dependencies only. The resulting models are simpler and easier to interpret when the networks are trained on different sample problems, including the recognition of regular languages and the computation of additions in different bases. 