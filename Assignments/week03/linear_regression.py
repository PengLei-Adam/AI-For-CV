## Linear Regression in pythonic way
###############################
import numpy as np
import random

def inference(w, b, X): 
    '''X is vector or matrix of m*n'''
    Y_pred = np.dot(X, w) + b
    return Y_pred

def eval_loss(w, b, X, gt_Y):
    return np.linalg.norm(np.dot(X, w) + b - gt_Y) * 0.5

def gradient(pred_Y, gt_Y, X):
    diff = pred_Y - gt_Y
    dw = np.mean(diff * X)
    db = np.mean(diff)
    return dw, db

def cal_step_gradient(batch_X, batch_gt_Y, w, b, lr):
    batch_pred_Y = inference(w, b, batch_X)
    dw, db = gradient(batch_pred_Y, batch_gt_Y, batch_X)
    w -= lr * dw
    b -= lr * db
    return w, b

def train(X, gt_Y, batch_size, lr, max_iter):
    w = 0
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
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    X = np.random.randn(num_samples) * 100
    Y = np.dot(X, w) + b + random.random() * random.randint(-1, 1)
    return X, Y, w, b
    
def run():
    X, Y, w, b = gen_sample_data()
    lr = 0.0001
    max_iter = 10000
    train(X, Y, 50, lr, max_iter)
    
    
if __name__ == '__main__':
    run()