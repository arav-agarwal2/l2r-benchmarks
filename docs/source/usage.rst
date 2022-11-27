Usage Instructions
==================

Welcome to l2r-benchmarks! This is a generic repo that aims to be not only a modularized implementation of RL and SafeRL algorithms,
but also an easier quickstart for those using l2r for their research. 

By using the interfaces we provide, we hope to make it easier for future research in SafeRL, and to empower future research efforts.


Setup
=====

In order to use l2r-benchmarks as-is, you will need to do the following:

Install the Arrival Simulator
-----------------------------

Download the Arrival Simulator from `the AiCrowd page <https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge>`__, 
signing the appropriate agreements. 

Confirm that your simulator runs on your target hardware, by trying to run it once.

Note: You cannot run the simulator from sudo, so you need to use ``sudo -u USERNAME ArrivalSim.sh`` to start the sim instead.

Additional Note: If you're a member of CMU and have access to Phoebe, ask us about our Kubernetes configurations, which should contain needed dependencies and work as-is.


Install the L2R framework
-------------------------

Besides the simulator, you will need access to `the l2r framework <https://github.com/learn-to-race/l2r>`__, which acts as a bridge between our python code and the simulator through a gym-like interface.


Currently, the best way to install this is through ``pip install git+https://github.com/learn-to-race/l2r@aicrowd-environment`` on the local environment. 
      
The ``aicrowd-environment`` branch aims to implemement changes from the aicrowd challenge into the current environment.

If you want to use the environment before this change, it is suggested to use ``pip install l2r`` instead, though this is all subject to change.



Install our requirements
------------------------

Install the python requirements using the command ``pip install -r setup/devtools_reqs.txt``

Note that we do require usage of some more recent typing features for configuration, so either use 3.8 or the typing_extensions requirement.


Download the sample VAE
------------------------

Download the sample VAE encoder from `here <https://drive.google.com/file/d/1Fe2ShIj16rkFb2qifBgtJx88Tj2xg6b0>`__.

If you're running from the terminal, ``pip install gdown && gdown 1Fe2ShIj16rkFb2qifBgtJx88Tj2xg6b0`` should work.


Running the codebase
====================

General codebase structure
--------------------------

``l2r-benchmarks`` contains the following file-structure:

| ├── ``Makefile``
| ├── ``README.md``
| ├── ``config_files``, which contains the configuration information for each run.
| ├── ``docs``, which contains the documentation information.
| ├── ``scripts``, which runs our agents.
| ├── ``src``, which contains the code for our agents.
| └── ``tests``, which contains some light and deprecated tests for our agents.


For most purposes, it is sufficient to simply edit the ``config_files``, ``src``, and ``scripts`` folders of ``l2r-benchmarks``.


Run with the existing codebase and config_files
-----------------------------------------------

Each example run is runnable using a different sub-folder of ``config_files``. For example, ``config_files/example_sac`` contains the parameters
necessary to run a simple run with our Soft Actor-Critic implementation. To change the config folder being used, simply change which ``runner.yaml`` file is being used
to construct the corresponding Runner object in ``scripts/main.py``.

As-is, the codebase will attempt to run the ``example_sac`` run. However, due to file-system differences, you will need to change the following parameters:

- Under ``runner.yaml``, edit the ``experiment_state_path`` and ``model_save_dir`` parameters, which will be where we store the interim experiment state and model parameter files.
- Under ``encoder.yaml``, edit the ``load_checkpoint_from`` path, which will be the location of the VAE file you downloaded earlier.

Once you do that, run the simulator using ``/PATH/TO/ArrivalSim.sh -OpenGL``, and run the experiment with ``python -m scripts.main``.


Contributing to the codebase
============================

If you want to do more than simply editing the codebase, please click :doc:`here </contributing>` for more information.