
'''
TODO: test
'''

import cv2
import random
import numpy as np

def ransacMatching(A, B):
    '''A & B: List of List, return homograpy matrix'''
    if len(A) != len(B):
        raise ValueError('Nums of A & B are different')
    randA, randB = A.copy(), B.copy()
    random.shuffle(randA)
    random.shuffle(randB)
    inlierA, outlierA = randA[:4], randA[4:]
    inlierB, outlierB = randB[:4], randB[4:]
    points_num = len(A)
    iter_num = 0
    iter_threshold = 100
    match_threshold = 50

    while True:
        iter_num += 1
        # Get the homography of the inliers
        homography = cv2.findHomography(inlierA, inlierB)
        match_pairs = []
        # Test all the other outliers
        for i, a in enumerate(outlierA):
            for j, b in enumerate(outlierB):
                match = np.linalg.norm(np.array(b) - np.dot(homography, np.array(a)))
                if j == 0:
                    min_match = match
                    min_idx = j
                if match < min_match:
                    min_match = match
                    min_idx = j
            if min_match < match_threshold:
                match_pairs.append((i+len(inlierA), min_idx+len(inlierA)))
        # Jump out
        if len(match_pairs) == 0 or iter_num > iter_threshold:
            break
        # Get new inliers and outliers
        in_pairs = [(i, i) for i in range(len(inlierA))] + match_pairs
        inlierA = [randA[p[0]] for p in in_pairs]
        inlierB = [randB[p[1]] for p in in_pairs]
        in_idx_A, in_idx_B = tuple(zip(*in_pairs))
        outlierA = [randA[i] for i in range(points_num) if i not in in_idx_A]
        outlierB = [randB[i] for i in range(points_num) if i not in in_idx_B]
        randA = inlierA + outlierA
        randB = inlierB + outlierB
        
    return homography




