import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from analysis_functions import *

def initiate_data_structure():
    # Where figures and data files are saved..
    PROJECT_ROOT_DIR = "Results"
    FIGURE_ID = "Results/FigureFiles"
    DATA_ID = "DataFiles/"

    if not os.path.exists(PROJECT_ROOT_DIR):
        os.mkdir(PROJECT_ROOT_DIR)
    if not os.path.exists(FIGURE_ID):
        os.makedirs(FIGURE_ID)
    if not os.path.exists(DATA_ID):
        os.makedirs(DATA_ID)
    def image_path(fig_id):
        return os.path.join(FIGURE_ID, fig_id)
    def data_path(dat_id):
        return os.path.join(DATA_ID, dat_id)
    def save_fig(fig_id):
        plt.savefig(image_path(fig_id) + ".png", format='png')
        
    print("Initiating data structure")
    print("Creating Files and Folders")
    print("")
    
# The Franke Function
def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

# Creates the design matrix
def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x) 
        y = np.ravel(y)
    N = len(x)
    l = int((n+1)*(n+2)/2) # Number of elements in beta
    X = np.ones((N,l))
    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
    return X
    
def create_dataset(poly):
    # Using a seed to ensure that the random numbers are the same everytime we run
    # the code. Useful to debug and check our code.
    # seed 3155
    np.random.seed(4242)
    # The degree of the polynomial (number of features) is given by
    n = poly #5
    # the number of datapoints
    N = 1000 # Sample size
    # the highest number of polynomials
    maxPolyDegree = 16
    # number of bootstraps
    n_bootstraps = 100
    # lambda values
    nlambdas = 10
    lambdas = np.logspace(-6,5, nlambdas)
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    # Remember to add noise to function 
    z = FrankeFunction(x, y) + 0.25*np.random.rand(N)
    X = create_X(x, y, n=n)
    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)
    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    
    y_train -= np.mean(y_train)
    y_test -= np.mean(y_test)
    
    error = np.zeros(maxPolyDegree)
    bias = np.zeros(maxPolyDegree)
    variance = np.zeros(maxPolyDegree)
    polynomial = np.zeros(maxPolyDegree)    
    return X_train,X_test,y_train,y_test,lambdas,maxPolyDegree,x,y,z, \
        error,bias,variance,polynomial,n_bootstraps,nlambdas