.. Covid-19 Impact Lab Simulator documentation master file, created by
   sphinx-quickstart on Thu Apr  9 19:42:05 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SID
==============

SID is a **S**\ imulator for **I**\ nfectious **D**\ iseases. It combines features of a prototypical Susceptible-Exposed-Infected-Recovered (SEIR) model and an agent based simulation model to analyze the spread of covid-19.

SID has only one public function:


.. automodule:: sid.simulate
    :members: simulate


The arguments of the function are explained in more detail in the reference guide:


.. toctree::
   :maxdepth: 1
   :caption: Reference Guide:

   states
   contact_models
   policies
