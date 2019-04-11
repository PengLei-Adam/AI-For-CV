
import itertools
import numpy as np

def medianBlur(img, kernel, padding_way):
    '''img & kernel is List of List; padding_way a string of "REPLICA" or "ZERO"'''

    def medianInCrop(crop):
        
        crop = list(itertools.chain.from_iterable(crop))
        return sorted(crop)[len(crop)//2]
    
    # padding
    W, H = len(img[0]), len(img)
    m, n = len(kernel[0]), len(kernel)
    w_pad, h_pad = m//2, n//2

    # different padding way
    if padding_way == 'REPLICA':
        img_pad = [[line[0]] * w_pad + line + [line[-1]] * w_pad for line in img]
        img_pad = [img_pad[0]] * h_pad + img_pad + [img_pad[-1]] * h_pad
    elif padding_way == 'ZERO':
        img_pad = [[0] * w_pad + line + [0] * w_pad for line in img]
        img_pad = [[0] * (w_pad * 2 + W)] * h_pad + img_pad + [[0] * (w_pad * 2 + W)] * h_pad
    else:
        raise ValueError('Input padding_way should be ""REPLICA" or "ZERO"')
    print(img_pad)

    using_np = True
    if using_np:
        img_array = np.array(img_pad, dtype=np.int)
        out_img = np.zeros((H, W), dtype=np.int)
        # select crop
        for i in range(H):
            for j in range(W):
                crop = img_array[i:i+n, j:j+m]
                out_img[i, j] = np.median(crop.reshape((m*n,)))
        return out_img.tolist()

    else:
        out_img2 = [[] for i in range(H)]
        for i in range(H):
            for j in range(W):
                crop = [line[j: j+m] for line in img_pad[i:i+n]]
                out_img2[i].append(medianInCrop(crop))
        return out_img2
    

if __name__ == "__main__":
    img = [[x for x in range(i, i + 5)] for i in range(5)]
    kernel = [[1,1,1]] * 3

    print(medianBlur(img, kernel, 'ZERO'))


    