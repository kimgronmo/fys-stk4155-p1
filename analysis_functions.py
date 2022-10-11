import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl # lasso regression

# imports resample from sklearn
# used in bootstrap
from sklearn.utils import resample


# import for cross validation
# should use 5-10 splits
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# imports Ridge regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

from support_functions import *

def R2(y_data, y_model):
    return 1 - ( (np.sum((y_data - y_model) ** 2)) / (np.sum((y_data - np.mean(y_data)) ** 2)) )


def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n
    
    
def SVDinv(A):
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    '''
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.pinv(D)
    return np.matmul(V,np.matmul(invD,UT))
    
    
# Start by defining the different reggression functions:
# OLS, Ridge, Lasso

# Ordinary Linear Regression
# returns beta values and checks against sklearn for errors
def OLS(xtrain,ytrain):
    # Testing my regression versus sklearn version
    # svd inversion ols regression
    OLSbeta_svd = SVDinv(xtrain.T @ xtrain) @ xtrain.T @ ytrain
    return OLSbeta_svd

# Ridge Regression
# returns beta values  
def RidgeManual(xtrain,lmb,identity,ytrain):
    Ridgebeta = SVDinv((xtrain.T @ xtrain) + lmb*identity) @ xtrain.T @ ytrain
    return Ridgebeta
    
    
# Generates and prints the Confidence Intervals for the betavalues
def betaConfidenceIntervals(X_train,betaValues,y_train,ytilde):

    print("\nCalculating and printing the Confidence Intervals for Beta:")

    # Calculating sample variance
    # N-p-1 for unbiased estimator, (3.8) from Hastie..
    # am I using correct p?
    sampleVar = np.sum((y_train-ytilde)**2)/(ytilde.size- betaValues.size- 1)

    # the variances are the diagonal elements
    betaVariance = np.diag(SVDinv(X_train.T @ X_train))
    sDev = np.sqrt(sampleVar)
    
    # For a 95% confidence interval the z value is 1.96
    zvalue = 1.96
    confidence = sDev*zvalue*np.sqrt(betaVariance)

    print("Lower bound:    Beta Values:    Upper bound:")    
    for i in range(betaValues.size):
        lower = betaValues[i] - confidence[i]
        upper = betaValues[i] + confidence[i]
        print(lower, betaValues[i], upper)
        
        
# bootstrap
def bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,regression,lambda1):
    if(regression == "OLS"):
        #print("We are doing OLS regression")
        for i in range(n_bootstraps):
            x_, y_ = resample(X_train,y_train)
            #print("trying first loop")
            # fits the data on same test data each time
            OLSbeta_svd_scaled = SVDinv(x_.T @ x_) @ x_.T @ y_
            y_pred[:,i] = (X_test @ OLSbeta_svd_scaled).ravel()
            y_pred[:,i] = (y_pred[:,i].T).ravel()
    if(regression == "Ridge"):
        #print("We are doing Ridge regression")
        sizeofX = X_train.shape
        tempmatrix = (X_train.T @ X_train)
        Identity = np.eye(tempmatrix.shape[0])
        
        for i in range(n_bootstraps):
            #print("bootstrap: ",i," of ",n_bootstraps)
            x_, y_ = resample(X_train,y_train)
            sizeofX = x_.shape
            tempmatrix = (x_.T @ x_)
            I = np.eye(tempmatrix.shape[0])
            
            Ridgebeta = RidgeManual(x_,lambda1,I,y_)
            y_pred[:,i] = (X_test @ Ridgebeta).ravel()
            y_pred[:,i] = (y_pred[:,i].T).ravel()
           

    if(regression == "LASSO"):
        print("We are doing LASSO regression")

    return y_pred
    
    
# Studying bias and variance trade-off as a function of model complexity
def biasVariance(regression,maxPolyDegree,polynomial,x,y,z,n_bootstraps,error, \
    bias,variance,nlambdas,lambdas):
    from support_functions import create_X
    
    # for use in ridge regressions
    min_mse_vals = []
    
    for polydegree in range(1, maxPolyDegree):
        polynomial[polydegree] = polydegree
        X = create_X(x, y, n=polydegree)
        # split in training and test data
        # assumes 75% of data is training and 25% is test data
        X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

        # something wrong with train test split when splitting..
        y_test.shape = (y_test.shape[0], 1)

        # scaling the data
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)
        
        y_train -= np.mean(y_train)
        y_test -= np.mean(y_test)        
        

        y_pred = np.empty((y_test.shape[0], n_bootstraps))    

        #Lets do the bootstrap analysis
        if(regression=="OLS"):
            y_pred = bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,"OLS",0)
        if(regression=="Ridge"):
            print("Starting bootstrap for polynomial: ",polydegree)
            # needs y_pred for each lambda
            counter1=0
            y_pred_lambda = []
            mse_valuesRidge=np.zeros(nlambdas)

            best_y_pred = np.zeros(len(y_train))
            min_mse = 100000.0
            
            for lmb in lambdas:
                y_pred = bootstrap(y_pred,X_train,X_test,y_train,n_bootstraps,"Ridge",lmb)
                y_pred_lambda.append(y_pred)
                
                # calculating mse dependent on lambda
                mse_valuesRidge[counter1]= MSE(y_test,y_pred)
                
                # finds the lowest mse for a given polynomial for all lambdas:
                if (mse_valuesRidge[counter1] <= min_mse):
                    #print("found value")
                    best_y_pred = y_pred
                    min_mse = mse_valuesRidge[counter1]
                counter1 +=1
            min_mse_vals.append(min_mse)
 
            # saves the best y_pred for a polynomial for all lambdas
            if(regression=="Ridge"):
                #print("found best value")
                y_pred = best_y_pred

            
            # use this to determine best fit for ridge?? for lambda and polynomial.
            # print best fit Ridge
            print("MSE values for Ridge regression with bootstrap")
            print(mse_valuesRidge)



        # Computing values for each polynomial
        polynomial[polydegree] = polydegree
        error[polydegree] = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias[polydegree] = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance[polydegree] = np.mean( np.var(y_pred, axis=1, keepdims=True) )

        
        if(regression=="OLS"):
            print('Polynomial degree:', polydegree)
            print('Error:', error[polydegree])
            print('Bias^2:', bias[polydegree])
            print('Var:', variance[polydegree])
            print('{} >= {} + {} = {}'.format(error[polydegree], bias[polydegree], variance[polydegree], bias[polydegree]+variance[polydegree]))

    
    error = error[1:]
    bias = bias[1:]
    variance = variance[1:]
    
    
    if(regression=="OLS"):
    # Plotting data for exercise2)
        plt.figure(2)
        plt.title("Bias-variance tradeoff as a function of model complexity, OLS regression", fontsize = 10)    
        plt.xlabel(r"Degree of polynomial sample size 1000", fontsize=10)
        plt.ylabel(r"mean square error", fontsize=10)
        plt.plot(polynomial[1:], error, label = "Error (MSE)")
        plt.plot(polynomial[1:], bias, label = "bias")
        plt.plot(polynomial[1:], variance, label = "variance")        
        
        plt.legend([r"mse from training data",r"bias from training data",r"variance from training data"], fontsize=10)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles', 'Ex c bias-var tradeoff.png'), transparent=True, bbox_inches='tight')
        plt.clf()

    if (regression=="Ridge"):
        #print("min vals mse ridge")
        #print(min_mse_vals)
        error = min_mse_vals

    # returns error to compare with cross validation
    return error

# Cross Validation
def crossValidation(polynomial,regression,x,y,z):
    #  Cross-validation code
    print("\nDoing cross-validation for polynomial: ",polynomial)
    from support_functions import create_X
    
    n = polynomial
    X = create_X(x, y, n=n)

    # split in training and test data
    # assumes 75% of data is training and 25% is test data
    X_train, X_test, y_train, y_test = train_test_split(X,z,test_size=0.25)

    # something wrong with train test split when splitting..
    y_test.shape = (y_test.shape[0], 1)

    # scaling the data
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    # lambda values
    nlambdas = 10
    lambdas = np.logspace(-6,5, nlambdas)

    # Initialize a Kfold instance
    k = 8
    kfold = KFold(n_splits = k)

    # Perform the cross-validation to estimate MSE
    scores_KFold = np.zeros((nlambdas, k))

    mse_values = np.zeros(k)

    counter = 0
    for train_inds, test_inds in kfold.split(X,z):
        X_train = X[train_inds]
        ytrain = z[train_inds]
        X_test = X[test_inds]
        ytest = z[test_inds]

        # scaling the data as usual
        X_train -= np.mean(X_train)
        X_test -= np.mean(X_test)

        # checking to see if I have to reshape y
        y_test.shape = (y_test.shape[0], 1)

        if(regression == "OLS"):
            OLSbeta_svd_scaled = SVDinv(X_train.T @ X_train) @ X_train.T @ ytrain
            y_pred = (X_test @ OLSbeta_svd_scaled).ravel()
        if(regression == "Ridge"):
            #print("test")
            # for each lambda make an ypred??
            as1=1 # indentation error

        # calculate mse for each split, then sum and average. Compare
        # with earlier code
        if(regression == "OLS"):
            mse_values[counter]=MSE(ytest,y_pred)

        counter += 1
    if (regression == "OLS"):
        print("Average MSE calculated from OLS cross-validation: ",np.mean(mse_values))
    return np.mean(mse_values)    
        
def LassoRegression(X_train,X_test,lambdas,y_train,y_test,fromSource,nlambdas):

    MSEPredictLasso = np.zeros(nlambdas)
    MSEPredictRidge = np.zeros(nlambdas)
    lambdas = np.logspace(-4, 0, nlambdas)

    for i in range(nlambdas):

        # for ridge regression
        sizeofX = X_train.shape
        tempmatrix = (X_train.T @ X_train)
        Identity = np.eye(tempmatrix.shape[0])
        Ridgebeta = RidgeManual(X_train,lambdas[i],Identity,y_train)
        
        y_predict = (X_test @ Ridgebeta).ravel() 
        lmb = lambdas[i]
        clf_lasso = skl.Lasso(alpha=lmb,tol=0.0001,max_iter=10000000).fit(X_train, y_train)
        ylasso = clf_lasso.predict(X_test)
        MSEPredictLasso[i] = MSE(y_test,ylasso)
        MSEPredictRidge[i] = MSE(y_test,y_predict)
    
    # Finds the minimum values calculated
    if (fromSource == "summary"):
        mse_min_LASSO = 100.0
        mse_minRidge = 100.0
        print("Minimum values for polynomial: 5 are: ")
        result = np.where(MSEPredictLasso == np.amin(MSEPredictLasso))
        print('Minimum MSE values for LASSO :', MSEPredictLasso[result[0]], "for lambda: ",lambdas[result[0]])
        result = np.where(MSEPredictRidge == np.amin(MSEPredictRidge))
        print('Minimum MSE values for Ridge :', MSEPredictRidge[result[0]], "for lambda: ",lambdas[result[0]])        
        
    #then plot the results
    plt.figure(5)
    plt.title("MSE Lasso versus Ridge", fontsize = 10)       
    plt.plot(np.log10(lambdas), MSEPredictRidge, 'r--', label = 'MSE Ridge Test')
    plt.plot(np.log10(lambdas), MSEPredictLasso, 'g--', label = 'MSE Lasso Test')
    plt.xlabel('log10(lambda)')
    plt.ylabel('MSE')
    plt.legend()
    
    if (fromSource == "realworld"):
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles',
        'MSE LASSO vs Ridge terrain data.png'), transparent=True, bbox_inches='tight')
    if (fromSource == "franke"):    
        plt.savefig(os.path.join(os.path.dirname(__file__), 'Results/FigureFiles',
        'MSE LASSO vs Ridge franke function.png'), transparent=True, bbox_inches='tight')
    plt.clf()
    
    # finds the lowest mse and returns it
    result = np.where(MSEPredictLasso == np.amin(MSEPredictLasso))
    lowest_mse = MSEPredictLasso[result[0]]
    #print(lowest_mse)
    return lowest_mse[0]
    
    
def summaryRegression(X_train,X_test,lambdas,y_train,y_test,nlambdas):
    print("\nSummary of regression\n")
    LassoRegression(X_train,X_test,lambdas,y_train,y_test,"summary",nlambdas)            
    betaValues = OLS(X_train,y_train)
    ytilde = X_train @ betaValues
    ypredict = X_test @ betaValues
    print("Minimum MSE values for OLS : ",MSE(y_test,ypredict)) 
    
    