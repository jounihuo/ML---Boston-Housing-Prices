## This code has been originally downloaded as a part of Udacity nanodegree.
## It has been modified by Jouni Huopana 15th of Nov 2015, in order to answer
## to posed questions. All modifications are for test use only.

"""Load the Boston dataset and examine its target (label) distribution."""
# version 0.2
# increased test sample size from 0.1 to 0.25
# added a histogram plot for the housing price
# version 0.3
# Corrections on the scoring method used


# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn import grid_search

## Nicer format for floats in commad line
## Source for the formating code 
## http://stackoverflow.com/questions/21008858/formatting-floats-in-a-numpy-array 
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

## For saving figures set fig_save = 1
fig_save = 1

def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data?
    
    nrow, ncol = housing_features.shape
    print("\n*******************************************************\n")
    print("  The Boston city data has the following properties:")
    print("\n*******************************************************\n")
    print("There is %i rows in the data." % nrow)
    print("There is %i columns in the data." % ncol)
    print("Total of %i values." % housing_features.size)
    
    print(city_data.DESCR)
    
    # Minimum value?
    # Calculating the minimum value with the Numpy's min function
    tmin = np.min(housing_features)
    print("\nMinimum value of the whole dataset is %.2f." % tmin)
    # Calculating the column minimums with the Numpy's amin function
    cmin = np.amin(housing_features, axis=0)
    print("Column specific minimums are :")
    print(cmin)
    print("Housing price minimum is %.2f" % np.min(housing_prices))
    # Maximum Value?
    # Corresponding Numpy functions are used for the maximums, means, medians, 
    # means and standard deviations.
    tmax = np.max(housing_features)
    print("\nMaximum value of the whole dataset is %.2f." % tmax)
    cmax = np.amax(housing_features, axis=0)
    print("Column specific maximums are :")
    print(cmax)
    print("Housing price maximum is %.2f" % np.max(housing_prices))
    # Calculate mean?
    cmean = np.mean(housing_features, axis=0)
    print("\nColumn specific means are :")
    print(cmean)
    print("Housing price mean is %.2f" % np.mean(housing_prices))
    # Calculate median?
    cmed  = np.median(housing_features, axis=0)
    print("\nColumn specific medians are :")
    print(cmed)
    print("Housing price median is %.2f" % np.median(housing_prices))
    # Calculate standard deviation?
    cstd  = np.median(housing_features, axis=0)
    print("\nColumn specific standard deviations are :")
    print(cstd)
    print("Housing price standard deviation is %.2f" % np.std(housing_prices))
    
    #Plot price histogram
    n, bins, patches = pl.hist(housing_prices, 20, histtype='bar', label=['House price'])
    pl.xlabel('House price')
    pl.ylabel('Count')
    pl.legend()
    # Figure saving
    if fig_save==1:
        pl.savefig('hist.png')
    pl.show()
    

def performance_metric(label, prediction):
    """Calculate and return the appropriate performance metric."""

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    
    # Calculating mean square error for the prediction
    # Scikit has its own function, but own one writen for practice
    # mse = mean_squared_error(label, prediction)
    mse = (1./prediction.size)*sum(np.power((prediction-label),2))
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return mse


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into training and testing set."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    
    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    
    # Creating the train and test sets with scikit's train_teast_split function
    # It provides easy set split with random sets.
    # Originally test_size=0.1 according to feedback changed to 0.25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

    return X_train, y_train, X_test, y_test


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.linspace(1, len(X_train), 50)
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot learning curve graph
    learning_curve_graph(depth, sizes, train_err, test_err)


def learning_curve_graph(depth, sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""
    # depth also passed for more accurate plot titles
    
    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size with depth %i' % depth)
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    # Grid added for clarity
    pl.grid()
    # Figure saving
    if fig_save==1:
        pl.savefig('dt_d_%i.png' % depth)
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    # Grid added for clarity
    pl.grid()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    # Figure saving
    if fig_save==1:
        pl.savefig('comp.png')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find the best performance metric
    # should be the same as your performance_metric procedure
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    rmse_scorer = make_scorer(performance_metric, greater_is_better = False)
    
    # 2. Use gridearch to fine tune the Decision Tree Regressor and find the best model
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    #Grid searching the Tree regressors parameters
    parameters = {'max_features':[1, 2, 3, 4, 5, 6, 7, 8 ,9], 
                  'max_depth':[1, 2, 3, 4, 5 ,6 ,7 ,8, 9], 
	        'min_samples_leaf':[1,2,3,4],
	        'min_weight_fraction_leaf':[0.01,0.05,0.1,0.2,0.3],
	        'random_state':[123]}
    
    # Fit the learner to the training data
    # Default scorer for the DecisionTreeRegressor is mse
    reg = grid_search.GridSearchCV(regressor, parameters, cv=10, verbose=1, scoring=rmse_scorer)
    print "Final Model: "
    print reg.fit(X, y)
    
    # Printing the best model form the grid search
    print(reg.best_estimator_)
    
    # Use the model to predict the output of a particular sample
    x = np.array([11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13])
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)
    
    #Plotting the final fit with the all of the data
    y = reg.predict(X)
    pl.plot(city_data.target, y, 'bo', label = 'Train data')
    pl.grid()
    pl.title('Final model predicting the whole data')
    pl.xlabel('Target')
    pl.ylabel('Prediction')
    # Figure saving
    if fig_save==1:
        pl.savefig('final_model.png')
    pl.show()
    
    


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
