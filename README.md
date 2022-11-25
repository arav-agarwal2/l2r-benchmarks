# l2r-benchmarks

## Setup

### Requirements -

-   Have the Arrival Simulator downloaded
-   Have the l2r framework installed in the environment you're planning to run experiments from - <https://github.com/learn-to-race/l2r> (aicrowd-environment) branch
    -   Installation command - pip install git+https://github.com/learn-to-race/l2r@aicrowd-environment 
-   Python version 3.8 or higher
-   Install the python requirements using the command - 
    -   pip install -r /setup/requirements.txt
-   Download the sample VAE encoder from [here](https://drive.google.com/file/d/1Fe2ShIj16rkFb2qifBgtJx88Tj2xg6b0/view?usp=sharing).
    -   If you're running from the terminal, pip install gdown && gdown 1Fe2ShIj16rkFb2qifBgtJx88Tj2xg6b0 should work.

### Running with the existing Agents and Vision Encoders -

-   Look through the /src/agents/ folder for existing agents
-   Look through the /src/encoders/ folder for existing encoders
-   For configurations, we have an existing folder in the /config_files/ directory called example_sac
-   To run the SAC you would need to change log locations in multiple yaml files in the /config_files/example_sac/ directory
    -   Change the "model_save_path" and "record_dir" in the agent.yaml to a local location where you are running the repository
    -   If loading an existing model set the "load_checkpoint" to True and add the location of the model to the "checkpoint"
    -   For loading the encoder use the "load_checkpoint_from" for passing the location of the encoder (Current one uses a vae rest of the configs are associated with that)
    -   For changing to a different encoder you would need to change the "name" and the "config" based on the inputs required by that particular encoder in the /src/encoders/ directory
    -   In the runner.yaml set the "model_save_dir" and "experience_save_dir" to local locations
    -   In case of experiment.yaml change the "experiment_state_path" from ': '/mnt/SAC_VAE/experimentState.json' to 'local_dir/experiment_name/experimentState.json"
-   After the configuration changes, run the simulator in the background using ./ArrivalSim.sh -OpenGL command
-   Now you can start running the l2r-benchmarks repository using the command - 
    -   python -m scripts.main

### Building your own Agents and using for training -

-   In the src/agents/ folder add your own code for the new agent structure of the agent is described in the base.py file. 
-   Important part is how to incorporate the agent specific variables into the codebase - 
    -   For that you would need to create a new agent.yaml file
    -   Firstly create a new folder in config_files/ similar to the existing example_sac folder
    -   In the folder update the yamls based on what you are changing - 
        -   For Encoder - "encoder.yaml" 
        -   For RL agent - "agent.yaml"
    -   You can use example_sac and the code base in src/agents/SAC.py and src/encoders/vae.py for RL agent and encoder respectively
-   To make sure your changes take effect in the scripts folder under the main.py change the line - runner = create_configurable("config_files/example_sac/runner.yaml", NameToSourcePath.runner) to runner = create_configurable("config_files/YOUR_FOLDER/runner.yaml", NameToSourcePath.runner)
-   Also remember to update the runner.yaml in your folder as well with the correct locations
