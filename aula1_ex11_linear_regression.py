import numpy as np
import pylab as pylab

#Evaluate the linear regression
def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size
   
    predictions = X.dot(theta).flatten()
    sqErrors = (predictions - y) ** 2
  
    J = (1.0 / (2 * m)) * sqErrors.sum()

    return J

def gradient_descent(X, y, theta, alpha, parada):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size

    custo_atual = 9999999999
    i = 0
    J_history = np.zeros(shape=(4000, 1))
    while custo_atual > parada:
        i = i + 1
        predictions = X.dot(theta).flatten()
        
        
        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]
        
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
        
        custo_atual = compute_cost(X, y, theta)
        J_history[i, 0] = custo_atual

    print 'numero de iteracoes', i
    return theta, custo_atual, J_history

#Load the dataset
data = np.genfromtxt('height.csv', delimiter=',')

X = data[:, 0]
y = data[:, 1]

#Some gradient descent settings
alpha = 0.01
parada = 0.001
m = y.size


#Add a column of ones to X (interception data)
it = np.ones(shape=(m, 2))

it[:, 1] = X

#Initialize theta parameters
theta = np.zeros(shape=(2, 1))

pylab.plot( X, y, 'o')
pylab.title('Idade vs Altura')
pylab.xlabel('Idade')
pylab.ylabel('Altura')
pylab.show()

theta, custo, J_history = gradient_descent(it, y, theta, alpha, parada)

print '\nthetas', theta, '\ncusto', custo

def minha_funcao(x, theta):
	return theta[0][0] + theta[1][0]*x

print J_history
pylab.plot(J_history)
pylab.title('Erro reduzindo a cada iteracao')
pylab.xlabel('Iteracao')
pylab.ylabel('Erro')
pylab.show()

pylab.plot(X, minha_funcao(X, theta), X, y, 'o')
pylab.title('Idade vs Altura')
pylab.xlabel('Idade')
pylab.ylabel('Altura')
pylab.show()