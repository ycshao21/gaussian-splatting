import collections
import numpy as np
import struct
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pi=3.141592654
result=[] # [[centerPoint,R],[centerPoint,R]......]


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
# 读入点云坐标bin
def read_points3D_binary(path_to_model_file):
    points3D={}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            points3D[point3D_id] = xyz
    return points3D

# 读入点云坐标txt
def read_points3D_text(path):
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
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

    #d2y = np.gradient(np.gradient(density, stepSum), stepSum)

    xpoints = np.array(stepSum)
    ypoints = np.array(density)
    plt.plot(xpoints, ypoints, linewidth=2.0, color='b',)
    plt.title(str(centerPoint[0])+','+str(centerPoint[1])+','+str(centerPoint[2]))
    plt.xlabel('R')
    plt.ylabel('density')
    plt.show()

    #for i in range(len(d2y) - 1):
        #if d2y[i] < 0 and d2y[i + 1] > 0:
            #return(stepSum[i])
    #return -1


# kmeans
def kmeans_clustering(data, num_clusters):

    coordinates = np.array(list(data.values()))

    # 初始化KMeans并进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(coordinates)

    # 返回每个簇的中心坐标
    return kmeans.cluster_centers_

def main():

    for k in [1,2,3,4,5,6,7,8,9,10]:
        # 读数据
        point3D=read_points3D_text("C:/Users/86181/Desktop/txt/points3D.txt")

        #kmeans求中心点
        centers = kmeans_clustering(point3D, k)

        # 求半径
        for centerPoint in centers:
            result.append([centerPoint,(getR(point3D,centerPoint))])

        print([k,result])
        file = open('C:/Users/86181/Desktop/out.txt', "a")
        file.write(str(k)+"个聚类:"+str([k,result]));
        file.close()


if __name__ == '__main__':
    main()