# Machine Learning Engineer Nanodegree
## Capstone Project
Youness Assassi
February 17, 2018
Financial Stocks Portfolio Analysis and trading

##About this project
This is the final project for the Udacity Machine Learning Nanodegree.  I chose to focus on finance and stock trading for this project.  Here is how it is broken down. 

##Dependencies
This project requires Python 2.7.  I used [Anaconda](http://continuum.io/downloads) which contained all of the necessary libraries, but they can also be added individually if necessary:
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer. 

### Prediction
The prediction folder contains the prediction module that will generate an optimized portfolio then provide recommendation on wether to invest in a porfolio in a given date.  

In a terminal or command window, navigate to the top level project directory report (that contains this README) and type

```bash
python -m prediction.start
```  

Update the train and predict dates in the run method contained in the 'prediction/start.py' directory with your chosen dates to simulate a trading scenario in the past.  

### Common
The common folder contains common helper functions using across the project

### Data 
The data folder contains all the data retrieved as part of this project.

### Processing
The processing modules contains methods for retrieving stock prices from yahoo finance. 
In order to retrieve a fresh set of data using a new date range, please update the start_tinme and end_time with your chosen values in the run method contained in the 'processing/start.py' directory, then in a terminal or command window, navigate to the top level project directory report (that contains this README) and type

```bash
python -m processing.start
```  

Note that this task may take a long time to complete as it will be retriving stock data one ticker at a time with a time delay to avoid getting your transaction cancelled by Yahoo Finance API.

### Statistics
The statistics folder contains modules for calculating different financial statistics

### Prediction
The prediction folder contains modules for predicting stock prices and recommending portfolios

### Report
In the report file, I describe the analysis that I made along with my conclusion about using machine learning in Finance and Trading.

