import numpy as np
import matplotlib.pyplot as plt

from analysis_functions import *
from support_functions import *

def print_ex_b(x,y,z):
    print("\nEvaluating mean squared error, R2 and Betas for polynomial 1-5")
    maxPoly = 5
    numPoly = np.zeros(maxPoly)
    mse_test = []
    R2_test = []
    beta_vals = []
    for i in range(1,maxPoly+1):
        numPoly[i-1] = i
        X = create_X(x, y, n=i)

        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)   
        y_train -= np.mean(y_train)
        y_test -= np.mean(y_test)        

        betaValues = OLS(X_train,y_train)
        beta_vals.append(betaValues)

        ypredict = X_test @ betaValues 

        mse_test.append(MSE(y_test,ypredict))
        R2_test.append(R2(y_test,ypredict))

    # have mse and R2 and beta values scores for poly 1-5
    plt.figure(1)
    plt.title("Test MSE as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"mean square error", fontsize=10)
    plt.plot(numPoly, mse_test, label = "MSE test data")
    plt.legend([r"mse from test data"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'MSE poly1-5 sample 1000.png'), transparent=True, bbox_inches='tight')    
    plt.clf()

    plt.figure(1)
    plt.title("Test R2 as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"R2", fontsize=10)
    plt.plot(numPoly, R2_test, label = "MSE test data")
    plt.legend([r"R2 from test data"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'R2 poly1-5 sample 1000.png'), transparent=True, bbox_inches='tight')    
    plt.clf()

    plt.figure(1)
    plt.title("Beta values as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Betas", fontsize=10)
    plt.ylabel(r"Beta values", fontsize=10)
    for i in range(maxPoly):
        plt.plot(range(1,len(beta_vals[i])+1), beta_vals[i], label = "MSE test data")
    plt.legend([r"Polynomial 1",r"Polynomial 2",r"Polynomial 3",r"Polynomial 4",r"Polynomial 5",], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'Beta values poly1-5 sample 1000.png'), transparent=True, bbox_inches='tight')    
    plt.clf()

def print_ex_d(error,mse_cv,polys):
    plt.figure(1)
    plt.title("MSE bootstrap vs CV as a function of model complexity,bootstrap=100,folds=8", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"MSE", fontsize=10)
    plt.plot(polys, error, label = "MSE test data bootstrap")
    plt.plot(polys, mse_cv, label = "MSE test data CV")    
    plt.legend([r"MSE test data bootstrap",r"MSE test data CV"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'mse bootstrap vs cv.png'), transparent=True, bbox_inches='tight')    
    plt.clf()

def print_ex_e(error,mse_cv,error_ridge,polys):
    plt.figure(1)
    plt.title("MSE bootstrap vs CV and Ridge as a function of model complexity,bootstrap=100,folds=8", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"MSE", fontsize=10)
    plt.plot(polys, error, label = "MSE test data bootstrap")
    plt.plot(polys, mse_cv, label = "MSE test data CV")
    plt.plot(polys, error_ridge, label = "MSE test data Ridge with bootstrap")      
    plt.legend([r"MSE test data bootstrap",r"MSE test data CV",r"MSE test data Ridge with bootstrap"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'mse bootstrap vs cv vs ridge.png'), transparent=True, bbox_inches='tight')    
    plt.clf()

def print_ex_f(error_mse,error_old,error_ridge,x,y,z,maxPolyDegree,lambdas,source,nlambdas,polynomials):
    numPoly = np.zeros(maxPolyDegree)
    mse_test_f=[]
    for i in range(1,maxPolyDegree):
        # Trying to print a progress bar...
        #print("#",end=''))
        numPoly[i-1] = i
        X = create_X(x, y, n=i)

        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)   

        y_train -= np.mean(y_train)
        y_test -= np.mean(y_test)
        mse_test_f.append(LassoRegression(X_train,X_test,lambdas,y_train,y_test,source,nlambdas))
        
    #print(error_mse)
    #print(error_old)
    #print(error_ridge)
    #print(mse_test_f)
    plt.figure(1)
    plt.title("MSE bootstrap vs CV and Ridge as a function of model complexity,bootstrap=100,folds=8", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"MSE", fontsize=10)
    plt.plot(polynomials, error_mse, label = "MSE test data bootstrap")
    plt.plot(polynomials, error_old, label = "MSE test data CV")
    plt.plot(polynomials, error_ridge, label = "MSE test data Ridge with bootstrap")    
    plt.plot(polynomials, mse_test_f, label = "MSE test data LASSO")       
    plt.legend([r"MSE test data bootstrap",r"MSE test data CV",r"MSE test data Ridge with bootstrap",r"MSE test data LASSO"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'mse bootstrap vs cv vs ridge vs LASSO.png'), transparent=True, bbox_inches='tight')    
    plt.clf()    

    print("Summary of regression for Franke function:")
    min_mse=100
    poly=100
    for i in range(15):
        if (error_mse[i] <= min_mse):
            min_mse=error_mse[i]
            poly=polynomials[i]
    print("lowest mse for OLS: ",min_mse," for polynomial: ",poly)
    min_mse=100
    poly=100
    for i in range(15):
        if (error_old[i] <= min_mse):
            min_mse=error_old[i]
            poly=polynomials[i]
    print("lowest mse for OLS with CV: ",min_mse," for polynomial: ",poly)
    min_mse=100
    poly=100
    for i in range(15):
        if (error_ridge[i] <= min_mse):
            min_mse=error_ridge[i]
            poly=polynomials[i]
    print("lowest mse for Ridge: ",min_mse," for polynomial: ",poly)
    min_mse=100
    poly=100
    for i in range(15):
        if (mse_test_f[i] <= min_mse):
            min_mse=mse_test_f[i]
            poly=polynomials[i]
    print("lowest mse for LASSO: ",min_mse," for polynomial: ",poly)    



    


def figure211(x,y,z):
    # makes figure 2.11 from Hastie..
    print("...this might take a wile... change maxPoly to speed up..")
    maxPoly = 16
    numPoly = np.zeros(maxPoly)
    mse_train211=[]
    mse_test211=[]
    for i in range(1,maxPoly+1):
        # Trying to print a progress bar...
        #print("#",end=''))
        numPoly[i-1] = i
        X = create_X(x, y, n=i)

        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)   

        y_train -= np.mean(y_train)
        y_test -= np.mean(y_test)
        

        betaValues = OLS(X_train,y_train)
        ytilde = X_train @ betaValues
        ypredict = X_test @ betaValues 
        mse_train211.append(MSE(y_train,ytilde))
        mse_test211.append(MSE(y_test,ypredict))

    # Generates plot and saves it in folder
    plt.figure(1)
    plt.title("Test and training error as a function of model complexity", fontsize = 10)    
    plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
    plt.ylabel(r"mean square error", fontsize=10)
    plt.plot(numPoly, mse_train211, label = "MSE training")
    plt.plot(numPoly, mse_test211, label = "MSE test data")
    plt.legend([r"mse from training data", r"mse from test data"], fontsize=10)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', \
        'figure211 from Hastie sample 1000.png'), transparent=True, bbox_inches='tight')
    #plt.show()
    plt.clf()