import numpy as np
import math
import torch
from sklearn.cluster import KMeans
import os
pi=3.141592654

# 读入点云坐标txt
def read_points3D(gaussian_points):
    points3D = {}
    for i in range(gaussian_points.shape[0]):
        point3D_id = i
        xyz = np.array(gaussian_points[i, :], dtype=np.float64)
        points3D[point3D_id] = xyz
    return points3D

# getR
def getR(point3D,centerPoint):
    centerDistance=[]
    for value in point3D.values():
        centerDistance.append(math.sqrt((value[0]-centerPoint[0])*(value[0]-centerPoint[0])+(value[1]-centerPoint[1])*(value[1]-centerPoint[1])
                                        +(value[2]-centerPoint[2])*(value[2]-centerPoint[2])))
    centerDistance.sort()
    step=1
    stepSum=[]
    density=[]
    index=0
    while step<math.ceil(centerDistance[len(centerDistance)-1]):
        if centerDistance[index]<=step:
            index=index+1
        else:
            density.append((index)/(4*pi*step*step*step/3))
            stepSum.append(step)
            step=step+1

    for i in range(len(density) - 2):
        if density[i]*3/4>density[i+1]:
            return i+1
    return -1

# kmeans
def kmeans_clustering(data, num_clusters):

    coordinates = np.array(list(data.values()))

    # 初始化KMeans并进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(coordinates)

    # 返回每个簇的中心坐标
    return kmeans.cluster_centers_

def getCenterAndR(gaussian_points, num_clusters):
    resultCenters =[]
    resultR=[]

    # 读数据
    point3D=read_points3D(gaussian_points)
    #kmeans求中心点
    centers = kmeans_clustering(point3D, num_clusters)

    # 求半径
    for centerPoint in centers:
        R = getR(point3D, centerPoint)
        if R == -1:
            continue
        resultCenters.append(torch.tensor(centerPoint, dtype=torch.float64, device="cuda"))
        resultR.append(getR(point3D, centerPoint))

    return resultCenters, resultR

