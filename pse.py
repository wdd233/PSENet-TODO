import numpy as np
from collections import deque
import skimage.measure
def pse(kernals, min_area):
    kernal_num = len(kernals)#判断里面有几个kernel
    pred = np.zeros(kernals[0].shape, dtype='int32')#最终的预测输出
    
    label, label_num = skimage.measure.label(kernals[0], neighbors=4, return_num=True)#四邻域标签，返回
    
    for label_idx in range(1, label_num):#小范围区域看成噪声,把面积小于阈值像素的位置置为空
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0

    queue = deque()
    next_queue = deque()
    points = np.array(np.where(label > 0)).transpose((1, 0))#初始化points存储所有点坐标(x,y)
    
    for point_idx in range(points.shape[0]):#遍历每一个点
        x, y = points[point_idx, 0], points[point_idx, 1]#取出每一个点的坐标
        l = label[x, y]#取出点的label值
        queue.append((x, y, l))#将该点送入队列
        # queue.put((x, y, l))
        pred[x, y] = l#对该点赋值

    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    for kernal_idx in range(1, kernal_num ):#遍历每一个kernel
        kernal = kernals[kernal_idx].copy()
        while len(queue):
            (x, y, l) = queue.popleft()#点出队列
            is_edge = True
            for j in range(4):#邻域判断
                tmpx = x + dx[j]#四个邻域遍历
                tmpy = y + dy[j]
                if tmpx < 0 or tmpx >= kernal.shape[0] or tmpy < 0 or tmpy >= kernal.shape[1]:
                    continue
                if kernal[tmpx, tmpy] == 0 or pred[tmpx, tmpy] > 0:#p>0说明该点已经用过了
                    continue
                queue.append((tmpx, tmpy, l))
                # queue.put((tmpx, tmpy, l))
                pred[tmpx, tmpy] = l
                is_edge = False
            if is_edge:
                next_queue.append((x, y, l))
                # next_queue.put((x, y, l))
        
        # kernal[pred > 0] = 0
        queue, next_queue = next_queue, queue
        
        # points = np.array(np.where(pred > 0)).transpose((1, 0))
        # for point_idx in range(points.shape[0]):
        #     x, y = points[point_idx, 0], points[point_idx, 1]
        #     l = pred[x, y]
        #     queue.put((x, y, l))

    return pred

if __name__ == '__main__':
    x = np.zeros((3,3,3))
    y = np.ones((3,3,3))
    s1 = np.zeros((5, 5))
    s2 = np.zeros((5, 5))
    s3 = np.zeros((5,5))
    s1[[0, 0, 0, 0], [0, 1, 2, 3]] = 1
    s2[[2, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 2]] = 1
    s3[[1,1,1,1],[0,1,2,3]] = 1
    # com = np.concatenate((x,y,x,y),axis=2)
    kernels = np.stack((s1,s3, s2))
    print(kernels)
    pred = pse(kernels, 2)
    print(pred)