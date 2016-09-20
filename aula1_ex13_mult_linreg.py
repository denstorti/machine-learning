from numpy import loadtxt, zeros, ones, mean, std, arange, array
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel, array

#Evaluate the linear regression

def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta)

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):

        predictions = X.dot(theta)

        theta_size = theta.size

        for it in range(theta_size):

            temp = X[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

	    J_history[i, 0] = compute_cost(X, y, theta)
	    # if i > 0 and J_history[i-1,0] != J_history[i, 0]:
    	print 'J_history[i-1,0]', J_history[i-1,0] 
    	print 'J_history[i, 0]', J_history[i, 0]

    return theta, J_history

#Load the dataset
data = loadtxt('iris.data.mod.txt', delimiter=',')

X = data[:, :4]
y = data[:, 4]

# print X
# print 'y', y

#number of training samples
m = y.size

y.shape = (m, 1)

#Scale features and set them to zero mean
#x, mean_r, std_r = feature_normalize(X)
x = X

#Add a column of ones to X (interception data)
it = ones(shape=(m, 5))
it[:, 1:5] = x

# print 'it', it

#Some gradient descent settings
iterations = 100000
alpha = 0.001

#Init Theta and Run Gradient Descent
theta = zeros(shape=(5, 1))
print theta

#array com dados ja aprendidos
#theta[:,0] = [0.1919334, -0.10971924, -0.04422724, 0.22699873 ,0.60988518]
# print theta

theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print 'J_history', J_history
print 'THETAS', theta

plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
# show()

def minha_funcao(theta, m, q):
    return 215810.61679138 + 61446.18781361 * ((m- mean_r[0]) / std_r[0]) + 20070.13313796 * ((q - mean_r[1]) / std_r[1])

#Predict price of a 1650 sq-ft 3 br house
classification = array([1.0,6.7,3.0,5.2,2.3]).dot(theta)
print 'Predicted price of a 6.7,3.0,5.2,2.3 flower is: %f' % (classification)
