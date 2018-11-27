from __future__ import division
from collections import Counter
import math
import random

#From ISBN-10 149190142X 'Data Science from Scratch', by Joel Grus
###########

#First we need to define some vector methods, where the vectors are merely lists. In practice this is very slow, but it's better for learning.
#In the future is is optimal to use things like NumPy Arrays. 


def vector_add(v, w):
    """adds corresponding elements""" 
    return [v_i + w_i for v_i, w_i in zip(v, w)]
    
def vector_subtract(v, w):
    """subtracts corresponding elements""" 
    return [v_i - w_i for v_i, w_i in zip(v, w)]
    
def vector_sum(vectors):
    """sums all corresponding elements""" 
    return reduce(vector_add, vectors)
        
def scalar_multiply(c, v):
    """c is a scalar, v is a vector"""
    return [c * v_i for v_i in v]
    
def vector_mean(vectors):
    """compute the vector whose ith element is the mean of the ith elements of the input vectors"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))
    
def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n""" 
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
    

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n""" 
    return dot(v, v)
    
def magnitude(v):
    return math.sqrt(sum_of_squares(v))
    
def squared_distance(v, w):
    """(v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2""" 
    return sum_of_squares(vector_subtract(v, w))
        
def distance(v, w):
    return magnitude(vector_subtract(v, w))

#We can build upon these methods to implement similar methods for matrices (for which we use nested lists here)

def shape(A):
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 
    return num_rows, num_cols
    # number of elements in first row
    
def get_row(A, i): 
    return A[i]

def get_column(A, j): 
    return [A_i[j] for A_i in A]

def make_matrix(num_rows, num_cols, entry_fn):
    """returns a num_rows x num_cols matrix whose (i,j)th entry is entry_fn(i, j)"""
    return [[entry_fn(i, j) 
        for j in range(num_cols)]
        for i in range(num_rows)]

def is_diagonal(i, j):
    """1's on the 'diagonal', 0's everywhere else""" 
    return 1 if i == j else 0

#Following are some methods for utilizing probability & statistics.
def mean(x):
    return sum(x) / len(x)
        
def median(v):
    """finds the 'middle-most' value of v""" 
    #Secretly, there are ways to call a median function on unsorted data (and so they're more efficient), but this is a bit more obvious for exposition purpose.
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2
    if n % 2 == 1:
        # if odd, return the middle value 
        return sorted_v[midpoint]
    else:
        # if even, return the average of the middle values 
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2

def quantile(x, p):
    """returns the pth-percentile value in x""" 
    p_index = int(p * len(x))
    return sorted(x)[p_index]
    

def mode(x):
    """returns a list, might be more than one mode""" 
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.iteritems()
    if count == max_count]
    
def data_range(x):
    return max(x) - min(x)
    

def de_mean(x): #(Deviation from Mean)
    """translate x by subtracting its mean (so the result has mean 0)""" 
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]
    
def variance(x):
    """assumes x has at least two elements""" 
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)
    
def standard_deviation(x): 
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)
    
def covariance(x, y): 
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)
    
def correlation(x, y):
    stdev_x = standard_deviation(x) 
    stdev_y = standard_deviation(y) 
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y 
    else:
        return 0 # if no variation, correlation is zero
        
def uniform_pdf(x): 
    return 1 if x >= 0 and x < 1 else 0

def uniform_cdf(x):
    "returns the probability that a uniform random variable is <= x"
    if x < 0: return 0 
    elif x < 1: return x 
    else: return 1

def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))


def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001): 
    """find approximate inverse using binary search"""
    if mu!=0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
    low_z, low_p = -10.0, 0
    hi_z, hi_p = 10.0, 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2 
        mid_p = normal_cdf(mid_z) 
        if mid_p < p:
            low_z, low_p = mid_z, mid_p 
        elif mid_p > p:
            hi_z, hi_p = mid_z, mid_p 
        else:
            break 
    return mid_z

#Some of these are for specific random variables.

def bernoulli_trial(p):
    return 1 if random.random() < p else 0
    
def binomial(n, p):
    return sum(bernoulli_trial(p) for _ in range(n))

#def sum_of_squares(v):
   # """computes the sum of squared elements in v""" 
    #return sum(v_i ** 2 for v_i in v)

def difference_quotient(f, x, h): 
    return(f(x+h)-f(x))/h
    
def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)] # add h to just the ith element of v
        
    return (f(w) - f(v)) / h
    
def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]

def step(v, direction, step_size):
    """move step_size in the direction from v""" 
    return [v_i + step_size * direction_i for v_i, direction_i in zip(v, direction)]

def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]
    
def safe(f):
    """return a new function that's the same as f, except that it outputs infinity whenever f produces an error""" 
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f
    
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001): 
    """use gradient descent to find theta that minimizes target function"""
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)
    # set theta to initial value
    # safe version of target_fn
    # value we're minimizing
    while True:
        gradient = gradient_fn(theta)
        next_thetas = [step(theta, gradient, -step_size) for step_size in step_sizes]
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        # stop if we're "converging"
    if abs(value - next_value) < tolerance: 
        return theta
    else:
        theta, value = next_theta, next_value

def negate(f):
    """return a function that for any input x returns -f(x)""" 
    return lambda *args, **kwargs: -f(*args, **kwargs)
    
def negate_all(f):
    """the same when f returns a list of numbers"""
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]
    

def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001): 
    return minimize_batch(negate(target_fn),
                            negate_all(gradient_fn),
                            theta_0,
                            tolerance)

def in_random_order(data):
    """generator that returns the elements of data in random order""" 
    indexes = [i for i, _ in enumerate(data)] # create a list of indexes 
    random.shuffle(indexes) # shuffle them
    for i in indexes: # return the data in that order 
        yield data[i]

#For when you're doing things point by point and you'd rather not be stuck around a local extrema forever.
def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = zip(x, y)
    theta = theta_0
    alpha = alpha_0
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
  
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )
        if value < min_value:
            # if we've found a new minimum, remember it # and go back to the original step size 
            min_theta, min_value = theta, value 
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size iterations_with_no_improvement += 1
            alpha *= 0.9
                    # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
    return min_theta

def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01): 
    return minimize_stochastic(negate(target_fn),
                                negate_all(gradient_fn),
                                x, y, theta_0, alpha_0)
                                


#Example Usage based on using random integers. 
    
tolerance = 0.0000001
v = [random.randint(-10,10) for i in range(3)]
print(v)
while True:
    gradient = sum_of_squares_gradient(v) 
    next_v = step(v, gradient, -0.01)
    if distance(next_v, v) < tolerance:
        break
    v = next_v
print(v)