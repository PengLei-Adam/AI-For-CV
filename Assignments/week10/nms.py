'''
Non-Maximum Suppression
'''
def NMS(lists, thre):
    
    if not lists:
        print('No region input.')
        return None

    remain = []
    s_lists = sorted(lists, key=lambda x: x[4])
    
    while s_lists:
        select = s_lists[-1]
        remain.append(select)
        del s_lists[-1]

        s_lists = [s for s in s_lists 
                    if IOU(s, select) <= thre]
        
    return remain
        
def IOU(reg1, reg2):
    x1 = max(reg1[0], reg2[0])
    y1 = max(reg1[1], reg2[1])
    x2 = min(reg1[2], reg2[2])
    y2 = min(reg1[3], reg2[3])

    if x2-x1 < 0 or y2-y1 < 0:
        return 0

    s_in = (x2-x1) * (y2-y1)
    s1 = (reg1[2] - reg1[0]) * (reg1[3] - reg1[1])
    s2 = (reg2[2] - reg2[0]) * (reg2[3] - reg2[1])

    return s_in / (s1 + s2 - s_in)

if __name__ == '__main__':
    lists = [
        [10, 10, 20, 20, 0.9],
        [11, 12, 21, 22, 0.81],
        [15, 15, 25, 26, 0.7],
        [18, 18, 28, 28, 0.85],
        [20, 20, 30, 30, 0.9],
        [23, 25, 33, 35, 0.8],
    ]
    threshold = 0.1

    print(NMS(lists, threshold))
    