# This is a bit of code from a computational modelling class, where the assignment was to use 
# multi-dimensional scaling as a very basic form of machine learning.  The goal was to create 
# a 2-D space to plot psychological distances between different abstract concepts (sports, in 
# this case) by minimizing error (stress) by using the gradient descent function.  I was provided
# some method names in the form of a very basic template, but I wrote the code for all of the 
# methods.


import numpy as np
import matplotlib.pyplot as plt

similarities = np.loadtxt(open("similarities.csv", "rb"), delimiter=",", skiprows=1)




D = 2 
N = distances.shape[0] 
assert(distances.shape[1] == N and N==len(names)) # be sure we loaded as many items as we have names for

def dist(a,b):
    # returns Euclidean distance between the locations of a, b
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    
def stress(p):
    # Take a matrix of positions 'p' and return the error from comparing it to the psychological distances, called stress
    s = 0
    for i in range(len(p)):
        for j in range(len(p[i])):
            if i != j:
                s += (distances[i][j] - dist(p[i],p[j]))**2
                
    return s


def add_delta(p, i, d, delta):
    # adds a 'delta' value to p[i,d]
    v = np.array(p)
    v[i, d] += delta
    return v

def compute_gradient(p, i,d, delta = 0.001):
    # compute the gradient of the stress function for the [i,d] entry of a position matrix p
    return (stress(add_delta(p,i,d, delta))-stress(add_delta(p,i,d,delta*-1)))/(2*delta)

    

def compute_full_gradient(p):
    # returns a matrix of the gradient of stress at p for each [i,d] coordinate
    grad = np.zeros(shape=(N,D))
    for x in range(len(p)):
        grad[x] = [compute_gradient(p, x, 0), compute_gradient(p, x, 1)]
    return grad




def findDistMat(p):
    # returns a matrix of distances from the matrix of positions, to be used in comparison to the psychological distance matrix
    d = np.zeros(shape=(len(p), len(p)))
    for x in range(len(d)):
        for y in range(len(d[x])):
            d[x][y] = dist(p[x],p[y])
    return d
            
def plotStress():
    stresslist = []
    iterlist = [1,3, 5, 7, 10, 100, 200, 400, 500, 700, 1000]

    for i in range(len(iterlist)):
        pos = np.random.normal(0.0,1.0,size=(N,D))
        rate=0.05
        for steps in range(iterlist[i]):

            fg = compute_full_gradient(pos)
            g = fg*rate*-1
            pos = np.add(g, pos)
        stresslist.append(stress(pos))

    plt.figure(figsize=(9, 10))
    plt.plot(iterlist, stresslist)
    plt.xlabel('Iterations')
    plt.ylabel('Stress')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Stress Over Iterations')

def simpleMDS():
    # attempt to find optimimum position matrix given the pyschological distance
    
    # converting similarities to distances naively
    distances = np.asarray([np.sqrt(1-i) for i in similarities])
    
    # initializes a position matrix with random positions
    pos = np.random.normal(0.0,1.0,size=(N,D))
    # initializes learning rate
    rate=0.05
    names = ["football","baseball","basketball","tennis","softball","canoeing","handball","rugby","hockey","ice hockey","swimming","track","boxing","volleyball","lacrosse","skiing","golf","polo","surfing","wrestling","gymnastics"]


    for steps in range(1000):
        
        fg = compute_full_gradient(pos)
        g = fg*rate*-1
        pos = np.add(g, pos)
    
        
        
    x=[]
    y=[]
    for i in pos:
        x.append(i[0])
        y.append(i[1])   
    plt.figure(figsize=(9, 10))
    plt.scatter(x, y)
    plt.title('2-D MDS Similarities Plot, Using D=1-S Transformation')
    for i, names in enumerate(names):
        plt.annotate(names, (x[i], y[i]))


def testMDS():
    # attempt to find optimimum position matrix given the pyschological distance
    
    # converting similarities to distances 3 ways
    distancesList = [np.asarray([1-i for i in similarities]), np.asarray([np.sqrt(1-i) for i in similarities]), np.asarray([(1/i)-1 for i in similarities])]    
    caption = ['D=1-S','D=Sqrt(1-S)', 'D=(1/S)-1']
    stressList = []
    
    plt.figure(figsize=(9, 18))
    plt.suptitle("Plots Using Different Transformations")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)


    rate=0.05

    for n in range(3):
        names = ["football","baseball","basketball","tennis","softball","canoeing","handball","rugby","hockey","ice hockey","swimming","track","boxing","volleyball","lacrosse","skiing","golf","polo","surfing","wrestling","gymnastics"]

        pos = np.random.normal(0.0,1.0,size=(N,D))
        distances = distancesList[n]
        for steps in range(300):

            fg = compute_full_gradient(pos)
            g = fg*rate*-1
            pos = np.add(g, pos)

        x=[]
        y=[]
        for i in pos:
            x.append(i[0])
            y.append(i[1])
        plt.subplot(3,1, n+1)
        plt.title('2-D MDS Similarities Plot, Using '+caption[n]+' Transformation, Stress of '+str(round(stress(pos), 3)))
        plt.scatter(x, y)
        for i, names in enumerate(names):
            plt.annotate(names, (x[i], y[i]))
        
plotStress()
simpleMDS()
testMDS()