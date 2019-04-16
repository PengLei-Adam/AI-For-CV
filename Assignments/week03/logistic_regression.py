## Linear Regression in pythonic way
###############################
import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)/(1.0-sigmoid(z))
    
def inference(w, b, X): 
    '''X is vector or matrix of m*n'''
    z = np.dot(X, w) + b
    Y_pred = sigmoid(z)
    return Y_pred

def eval_loss(w, b, X, gt_Y):
    return np.linalg.norm(inference(w, b, X) - gt_Y) * 0.5

def gradient(gt_Y, X, w, b):
    z = np.dot(X, w) + b
    pred_Y = sigmoid(z)
    diff = pred_Y - gt_Y
    sp = sigmoid_prime(z)
    dw = np.mean((diff * sp * X.T).T, axis=0)
    db = np.mean(diff * sp)
    return dw, db

def cal_step_gradient(batch_X, batch_gt_Y, w, b, lr):
    dw, db = gradient(batch_gt_Y, batch_X, w, b)
    w -= lr * dw
    b -= lr * db
    return w, b

def train(X, gt_Y, batch_size, lr, max_iter):
    w = 0 if len(X.shape) == 1 else np.zeros((X.shape[1],))
    b = 0
    num_samples = X.shape[0]
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_X = X[batch_idxs]
        batch_gt_Y = gt_Y[batch_idxs]
        w, b = cal_step_gradient(batch_X, batch_gt_Y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, X, gt_Y)))

def gen_sample_data():
    num_samples = 100
    w = random.random()		# for noise random.random[0, 1)
    b = random.random()
    X = np.random.randn(num_samples)
    Y = sigmoid(np.dot(X, w) + b)+ np.random.randn(num_samples) * 0.1
    return X, Y, w, b
    
def run():
    X, Y, w, b = gen_sample_data()
    lr = 0.1
    max_iter = 10000
    train(X, Y, 50, lr, max_iter)
    print(w, b)
    
    
if __name__ == '__main__':
    run()