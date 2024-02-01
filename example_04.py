# from IPython import display
# display.Image("8143e0da24f78ea9d7e6.jpg")
import numpy as np
from autograd import grad
import autograd.numpy as np1
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import time
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

# RK4 method
def ode_solve_G(z0, G):
    """
    Simplest RK4 ODE initial value solver
    """
    n_steps = 500
    z = z0
    # print(z)
    h = np.array([0.05])
    for i_step in range(n_steps):
        k1 = h*G(z)

        k2 = h * (G((z+h/2)))
        k3 = h * (G((z+h/2)))
        k4 = h * (G((z+h)))
        k = (1/6)*(k1+2*k2+2*k3+k4)
        #k = k.reshape(1,10)
        z = np.array([z0]).reshape(10,1)
        z = z + k
        #print("z;",z.shape)
    return z
# def f(x):
#     return (x[0]**2 + x[1]**2 + 3) / (1 + 2*x[0] + 8*x[1])
def f(x):
    """
    Objective function f(x) as defined in the problem.
    x: vector of variables
    q: vector of parameters
    """
    # tmp = []
    # for i in range(10):
    #     tmp.append(np.random.rand(1).tolist()[0])
    tmp = 10*[1]
    # print(tmp)
    q = np1.array(tmp)
    return -np1.exp(-np1.sum((x**2) / (q**2)))
# def g1(x):
#     return -x[0]**2 - 2*x[0]*x[1] + 4
# def g2(x):
#     return -x[0]
# def g3(x):
#     return -x[1]

def g_i(x):
    i =1
    """
    Inequality constraint g_i(x) as defined in the problem with a sequence of squared terms.
    x: vector of variables
    i: the index of the inequality constraint function g_i
    """
    # Compute the sequence of squared terms
    # We adjust the indices for 0-based indexing used in Python.
    # The sequence is x_{10*(i-1)+1}^2 to x_{10*(i-1)+10}^2, hence we use a slice [10*(i-1):10*i]
    squared_terms = x[10*(i-1):10*i]**2

    # Compute the inequality constraint function value
    g_x_i = np1.sum(squared_terms) - 20
    
    return g_x_i
def derivative_g_i(x):
    i = 1
    """
    Derivative of the inequality constraint g_i(x) with respect to x, as defined in the problem.
    x: vector of variables
    i: the index of the inequality constraint function g_i
    """
    # Initialize the gradient as a zero vector of the same length as x
    grad = np.zeros_like(x)

    # Compute the gradient only for the terms involved in the ith inequality
    # Adjust for 0-based indexing: the terms are x_{10*(i-1)+1} to x_{10*(i-1)+10}
    indices = range(10*(i-1), 10*i)
    grad[indices] = 2 * x[indices]

    return grad
def g3(x):
    x = np.array(x)
    A = np.array([[1,1,1,1,1,3,3,3,3,3]])
    b = np.array([[16]])
    return (A@(x.T) - b.T).tolist()[0][0] # 
# g1_dx = grad(g1)
# g2_dx = grad(g2)
# g3_dx = grad(g3)
# g_dx = [g1_dx,g2_dx]
f_dx = grad(f)
# bounds = Bounds([0,0],[np.inf,np.inf])
cons = ({'type': 'eq',
          'fun' : lambda x: np.array([g3(x)]),
          'jac' : lambda x: np.array([1,1,1,1,1,3,3,3,3,3])},
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g_i(x)]),
          'jac' : lambda x: np.array([-grad(g_i)(x)])})
def rosen(x,y):
    """The Rosenbrock function"""
    return np.sqrt(np.sum((x-y)**2))
def find_min(y,n):
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point",hess=BFGS(),
                constraints=cons,method='trust-constr', options={'disp': False})
    return res.x
def run_nonsmooth1(x, max_iters, f, f_dx,n,alpha,mu0):
    res = []
    val = []
    lda = 1 #1e9
    sigma = 0.1 #100
    mut = mu0
    K = np.random.rand(1,1)
    res.append(x)
    val.append(f(x))
    x_pre = x
    for t in range(max_iters):
        y = x - lda*f_dx(x)
        x_pre = x.copy()
        x = find_min(y,n)
        if f(x) - f(x_pre) + sigma*(np.dot(f_dx(x_pre).T,x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K*lda
        #mut = mut*np.exp(-alpha*t)
        res.append(x)
        val.append(f(x))
    #print(x)
    return res,x
def Phi(s):
    if s > 0:
        return 1
    elif s == 0:
        return np.random.rand(1)
    return 0
# Neural network
A = np.array([[1,1,1,1,1,3,3,3,3,3]])
b = np.array([[16]])
def G(x):

   
    #g3x = g3(x)
    gx = [g_i(x)]
    g_dx = [grad(g_i)]
    c_xt = 1.
    Px = np.zeros((10, 1))

    for (i,j) in zip(gx, g_dx):
        c_xt *= (1-Phi(i))
        #print(Phi(i)*(j(x)))
        # print(j)
        # print(Phi(i))
        # print(Px)
        #print(Phi(i),j(x))
        # print("Px:",Px)
        # print(x)
        # print(np.array([Phi(i)*j(x)]))
        #print(x.shape)
        Px += np.array([Phi(i)*j(x)]).reshape(10,1)
    c_xt *= (1-Phi(np.abs(A@(x) - b)))
    
    eq_constr_dx = ((2*Phi(A@(x)-b)-1)*A.T)
    # print(-c_xt*f_dx(x))
    # print(Px)
    # print(eq_constr_dx)
    #print((np.array([-c_xt*f_dx(x)]).reshape(10,1) - Px - eq_constr_dx) .shape)
    #print(((2*Phi(A@(x)-b)-1)*A.T).shape)
    return np.array([-c_xt*f_dx(x)]).reshape(10,1) - Px - eq_constr_dx
def run_nonsmooth(x0, max_iters):
    xt = x0
    res = []
    res.append(xt.tolist())
    for t in range(max_iters):
        xt = ode_solve_G(xt,G)
        #print(xt.shape)
        #print(xt.shape)
        #print(xt.reshape(1,10).tolist())
        res.append(xt.reshape(1,10).tolist()[0])
    # print(xt)
    # print(f(xt))
    return res,xt.reshape(1,10)

def plot_x(sol_all,count,max_iters):
    t = [i for i in range(max_iters+1)]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    for i in range(count):
        if i ==0:
            text_color = 'red'
            text_label = r'$x_{1}(t)$'
        else:
            text_color = 'green'
            text_label = r'$x_{2}(t)$'
        plt.plot(t, sol_all[i][:,0],color=text_color,label=text_label,linewidth=1)
        plt.plot(t, sol_all[i][:,1],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,2],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,3],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,4],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,5],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,6],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,7],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,8],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,9],color=text_color,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    plt.legend([r'$x_{1}(t)$',r'$x_{2}(t)$']) #,r'$x_{4}(t)$',r'$x_{5}(t)$',r'$x_{6}(t)$',r'$x_{7}(t)$',r'$x_{8}(t)$',r'$x_{9}(t)$',r'$x_{10}(t)$'])
    plt.legend()
    plt.show()

def main_GDA(num, max_iters, n, alpha, mu0):
    sol_all1 = []
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)  # Start point
        _, xt = run_nonsmooth1(x0, max_iters, f, f_dx, n, alpha, mu0)
        print("Result GDA - x*:", xt)
        print("Value -ln(-f(x*)) of GDA:", -np.log(-f(xt)))
        sol_all1.append(xt)
    return sol_all1

def main_RNN(num, max_iters, n):
    sol_all = []
    for i in range(num):
        x0 = np.random.rand(1, n)
        x0 = find_min(x0, n)  # Start point
        _, xt = run_nonsmooth(x0, max_iters)
        print("Result RNN - x*:", xt)
        print("Value -ln(-f(x*)) of RNN:", -np.log(-f(xt)))
        sol_all.append(xt)
    return sol_all

if __name__ == '__main__':
    # Main parameters
    num = 1  # Number of starting points
    max_iters = 10  # Maximum number of iterations
    n = 10  # Size of x
    alpha = np.random.rand(1)  # Alpha parameter
    mu0 = np.random.rand(1)  # Mu0 parameter

    # Run the main function for GDA
    result_GDA = main_GDA(num, max_iters, n, alpha, mu0)

    # Run the main function for RNN
    result_RNN = main_RNN(num, max_iters, n)

    # Plot trajectory
    # plot_x(sol_all1,count,max_iters1)