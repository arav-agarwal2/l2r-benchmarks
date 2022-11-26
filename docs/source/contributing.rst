Contributing
============


So - you want to add to this codebase. To do so, you'll want a high-level flow on the different aspects of the codebase, and some idea of the 
configuration system. Under ``src``, you will notice the following folders:

| ├── ``agents``, which contain RL algorithm code.
| ├── ``buffers``, which contain replay buffer implementations.
| ├── ``config``, which contains the config parsing system.
| ├── ``encoders``, which contains the encoder implementations.
| ├── ``loggers``, which contains the logger implementations ( interfaces to tensorboard and wandb ).
| ├── ``networks``, which contains the network modules.
| ├── ``runners``, which contains the runner implementations.
| └── ``utils``, which contains some commonly used utility functions ( though it's mostly empty ).

To add to any of these sections, you'll want to follow the following flow:

- Implement / Transfer the algorithm, following the ``base.py`` file for the intended interface.
- Add configuration doc strings to the ``__init__`` function of the class.
- Edit the ``__init__.py`` file in the folder you're adding to to import the algorithm's class.
- Add the ``@yamlize`` decorator to the class.
- Create a new folder under ``config``, changing the internal references to ensure all files link properly.
- Edit the folder and try running the code to confirm that it works.
- Document it using Google-style docstrings, to update our documentation.

In particular, it pays to focus on the configuration side of this equation for a second.

To instantiate any class with the ``@yamlize`` decorator, you can use the ``create_configurable`` and ``create_configurable_from_dict`` functions. 
These take in a YAML/dict with the following interface::

  name: NAME_OF_CLASS
  config: CONFIG_DICT_OF_CLASS


Here, the name of the class is used to import the class from the specific folder using importlib, which is then instantiated using the config dict.

The configuration dict of the class comes from the type information you provide to the ``__init__`` function. The ``yamlize`` constructor takes in the type information,
and attempts to convert that into strictyaml validators to convert the strings to the intended output without any worry. Should this fail, you'll get an error.

To see what the YAML schema is for any class, either look at our module docs, or simply call ``CLASSNAME.schema``, which will print out the according schema information.

Should you want to create a class that creates another class upon instantiation, you can either:

- Pass a parameter pointing to another yaml file, which can then be loaded using ``create_configurable``
- Pass a parameter with the ``ConfigurableDict`` type, which you can then load using the ``create_configurable_from_dict`` function.

Note that optional parameters are respected by ``create_configurable``, so simply removing an item will set it to the default from the code. 

However - empty config dicts may not work, simply due to how ``create_configurable_from_dict`` functions currently.

