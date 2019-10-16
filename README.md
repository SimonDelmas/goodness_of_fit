# Goodness of fit 

goodness_of_fit is a python language software package that provide a set of function 
for goodness of fit measure between two signals.

While most of these functions are available in packages such as [Scipy](https://github.com/scipy/scipy), [Spotpy](https://github.com/thouska/spotpy), etc... 
this package brings together all these functions and provides a unified interface for their use.

## Content of the package

The package provides the following functions :
* Mean Error
* Mean Absolute Error
* Root Mean Square Error
* Normalized Root Mean Square Error
* Pearson product-moment correlation coefficient
* Coefficient of Determination
* Index of Agreement
* Modified Index of Agreement
* Relative Index of Agreement
* Ratio of Standard Deviations
* Nash-sutcliffe Efficiency
* Modified Nash-sutcliffe Efficiency
* Relative Nash-sutcliffe Efficiency
* Kling Gupta Efficiency
* Deviation of gain
* Standard deviation of residual

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

goodness_of_fit requires :

* Python 3
* [Numpy](https://github.com/numpy/numpy) for efficient computation on array.

### Installing

To install the package, clone or download the repository and use the setup.py :

```bash
git clone https://github.com/SimonDelmas/goodness_of_fit.git
cd goodness_of_fit
python ./setup.py install
```

### Building the documentation

The documentation could be generated using the command :

```bash
python ./setup.py build_sphinx
```
 

### Running the tests

After installation, you can launch the test suite with pytest :

```bash
pytest
```

## License

This project is licensed under the GLP-2.0 License - see the [LICENSE.md](LICENSE.md) file for details.
