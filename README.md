# Project of a neural network predicting prices of the S&P 500 stock index

The aim of the project was to design the best possible neural network model to forecast the prices of the S&P 500 stock index, using historical data from 1982 to 2023.

The main libraries used to design the network were TensorFlow and Keras

## Description

Steps taken in the project:

* EDA(Exploratory Data Analysis) - graphical analysis of the data, validation and creation of a valid dataset for training and testing of the neural network
* feature engineering - development of a function that creates financially relevant features from the dataset received
* scaling of features and creation of training and testing sets
* creation, compilation, training and evaluation of the initial model
* tuning the parameters of the neural network using the hparams module and Tensorboards
* creation, compilation, training and evaluation of the final model with the best params
* comparison of actual and predict values on the plot

## Getting Started

### Dependencies

* Linux-Ubuntu 20.04
* Python 3.9

### Installing

#### The first thing to do is to clone the repository:

```sh
$ git clone https://github.com/bartoszsklodowski/stock_values_NN.git
$ cd stock_values_NN
```

#### Create a virtual environment to install dependencies in and activate it:

```sh
$ python3 -m venv .venv
$ source .venv/bin/activate
```

#### Then install the dependencies:

```sh
(.venv)$ pip install -r requirements.txt
```
Note the `(.venv)` in front of the prompt. This indicates that this terminal
session operates in a virtual environment set up by `virtualenv`.


### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Author

Bartosz Skłodowski 
