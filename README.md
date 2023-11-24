# generatorpipeline
Parallelize your data-processing pipelines with just a decorator.


# Tutorial
Please see the [tutorial notebook](generatorpipeline-tutorial.ipynb) for an overview of the functionality.


# Installation
Python 2 is NOT supported. You must use __python version 3.6__ or higher! 

1) The recommended way is to create a python venv and install it into the venv. Create a new virtualenv by
```
python -m venv --system-site-packages ~/.venv/defaultpyvenv
```

2) activate the environment using `source ~/.venv/defaultpyvenv/bin/activate`.

3) Install into the venv
```
pip install generatorpipeline[full]@git+https://github.com/skuschel/generatorpipeline.git
```

# Installation for developers

Follow steps 1 and 2 of the normal installation to create and activate a venv.

3) git clone this repository
```
git clone git@github.com:skuschel/generatorpipeline.git
```

4) Install in editable mode using
```
pip install -e .
```



# Contributing
... is always welcome! Development and issue tracker can be found on github. Please report bugs to
https://github.com/skuschel/generatorpipeline/issues
