# -*- coding: utf-8 -*-

# programmer: xlxlqqq
# data:2022.5.16

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# 初始化输入
def init(path1,path2):
    img1 = cv.imread(path1,0)
    img2 = cv.imread(path2,0)

    K = np.array([[4678.430192307692, 0, 2136], 
        [0, 4678.430192307692, 1424],
        [0, 0, 1]])

    return img1, img2, K


# 使用SIFT寻找匹配点
def Match(img1,img2):

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # bf匹配和k近邻
    BFMatch = cv.BFMatcher()
    matches = BFMatch.knnMatch(des1, des2,k=2)

    return matches,kp1,des1,kp2,des2

    
#挑选准确的八组匹配点
def Select(matches,img1, kp1, img2, kp2):
    # global img1,img2,kp1,kp2

    WellMatch = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            WellMatch.append([m])

    WellMatch = sorted(WellMatch, key=lambda x: x[0].distance) 

    # 排除掉重复的点对,取出八个最优的
    for m in WellMatch: 
        for n in WellMatch:
            if  n!=m and (m[0].trainIdx == n[0].trainIdx):
                WellMatch.remove(n)
    WellMatch = WellMatch[:8] 

    img_match = cv.drawMatchesKnn(img1, kp1, img2, kp2, WellMatch, None, flags=2)
    cv.imwrite("./image/match.png", img_match)

    Positions = []
    for i in WellMatch:
        Positions.append([kp1[i[0].queryIdx].pt,kp2[i[0].trainIdx].pt])

    # 标注匹配点
    for i in range(8):
        color = (255,255,255)
        const1 = 4272
        Matchedpoints = Positions[i]
        handleIm1 = cv.circle(img_match,(int((Matchedpoints[0])[0]),int((Matchedpoints[0])[1])), 20, color, -1)
        handleIm2 = cv.circle(img_match,(int((Matchedpoints[1])[0]) + const1,int((Matchedpoints[1])[1])), 20, color, -1)
        handleText = cv.putText(img_match,str(i),(int((Matchedpoints[0])[0]),int((Matchedpoints[0])[1])),cv.FONT_HERSHEY_SIMPLEX,10,color,5)
        handleline = cv.line(img_match,(int((Matchedpoints[0])[0]),int((Matchedpoints[0])[1])),(int((Matchedpoints[1])[0]) + const1,int((Matchedpoints[1])[1])),color,5,-1)
    cv.imwrite("./image/match.png", img_match)

    return WellMatch,img_match,Positions



# 计算某矩阵的归一化矩阵
def GetNmMatr(px,py):
    u = sum(px)/8
    v = sum(py)/8
    s = np.sqrt(2)*8 / sum(np.sqrt((px-u)*(px-u)+(py-v)*(py-v)))
    T = np.array(([s,0,-s*u],[0,s,-s*v],[0,0,1]))
    return T

# 计算基础矩阵
def GetFdmtMatr(px1,py1,px2,py2):
    T1 = GetNmMatr(px1,py1) 
    T2 = GetNmMatr(px2,py2)
    Q1=(T1@np.row_stack((px1,py1,np.ones((1,8))))).T
    Q2=(T2@np.row_stack((px2,py2,np.ones((1,8))))).T
    Q=np.array([list(map(lambda x,y:x*y,Q1[:,0],Q2[:,0])),
                list(map(lambda x,y:x*y,Q1[:,1],Q2[:,0])),
                list(map(lambda x,y:x*y,Q1[:,2],Q2[:,0])),
                list(map(lambda x,y:x*y,Q1[:,0],Q2[:,1])),
                list(map(lambda x,y:x*y,Q1[:,1],Q2[:,1])),
                list(map(lambda x,y:x*y,Q1[:,2],Q2[:,1])),
                list(map(lambda x,y:x*y,Q1[:,0],Q2[:,2])),
                list(map(lambda x,y:x*y,Q1[:,1],Q2[:,2])),
                list(map(lambda x,y:x*y,Q1[:,2],Q2[:,2])),]).T

    U,D,VT = np.linalg.svd(Q)
    # 约束条件
    F = VT.T[:, 8].reshape(3, 3) 
    U1, D1, VT1 = np.linalg.svd(F) 
    F = U1 @ np.diag((D1[0], D1[1], 0)) @ VT1
    # 逆归一化
    F = T2.T @ F @ T1
    return F

# 计算得到旋转和平移矩阵
def SolveRT(E,which,Parray):
    Ue,De,VTe = np.linalg.svd(E)
    k=(De[1]+De[2])/2
    # 计算正交矩阵
    W=np.array(([0,-1,0],[1,0,0],[0,0,1]))
    # 计算其反对称阵
    Z=np.dot(np.diag([1,1,0]),W)

    R1 = Ue @ W.T @ VTe
    R2 = Ue @ W @ VTe
    T1 = Ue @ np.diag([0,0,1])
    T2 = Ue @ np.diag([0,0,-1])

    P11 = K @ np.append(R1,np.array([T1[:,2]]).T,axis=1)
    P12 = K @ np.append(R1,np.array([T2[:,2]]).T,axis=1)
    P21 = K @ np.append(R2,np.array([T1[:,2]]).T,axis=1)
    P22 = K @ np.append(R2,np.array([T2[:,2]]).T,axis=1)
    P0 = K @ np.append(np.eye(3),[[0],[0],[0]],axis=1)
    Pos1 = cv.triangulatePoints(P0, P11, Parray[:,0][which], Parray[:,1][which])
    Pos2 = cv.triangulatePoints(P0, P12, Parray[:,0][which], Parray[:,1][which])
    Pos3 = cv.triangulatePoints(P0, P21, Parray[:,0][which], Parray[:,1][which])
    Pos4 = cv.triangulatePoints(P0, P22, Parray[:,0][which], Parray[:,1][which])
    
    # 分类得到最终的RT和误差
    if Pos1[2]>0:
        return R1,T1,1
    elif Pos2[2]>0:
        return R1,T2,2
    elif Pos3[2]>0:
        return R2,T1,3
    elif Pos4[2]>0:
        return R2,T2,4
    else:
        return 0,0,0


# def Verify(Positions,R,T):
    # TransMatrix = [
    #                 [R[0,0],R[0,1],R[0,2],T[0,2]],
    #                 [R[1,0],R[1,1],R[1,2],T[1,2]],
    #                 [R[2,0],R[2,1],R[2,2],T[2,2]],
    #                 [0,0,0,1]]

#     Pos1k = TransMatrix .* Positions[0,1]
#     Pos2k = TransMatrix .* Positions[1,1]
#     Pos3k = TransMatrix .* Positions[2,1]
#     Pos4k = TransMatrix .* Positions[3,1]
#     Pos5k = TransMatrix .* Positions[4,1]
#     Pos6k = TransMatrix .* Positions[5,1]
#     Pos7k = TransMatrix .* Positions[6,1]
#     Pos8k = TransMatrix .* Positions[7,1]

#     return Pos1k,Pos2k,Pos3k,Pos4k,Pos5k,Pos6k,Pos7k,Pos8k




if __name__ == '__main__': 

    path1 = "./image/001.jpg"
    path2 = "./image/002.jpg"

    img1,img2,K = init(path1,path2)

    # print(img1)
    # print(img2)

    matches,kp1,des1,kp2,des2 = Match(img1,img2)
    # print(type(kp1))

    WellMatch, img_match, Positions = Select(matches,img1, kp1, img2, kp2)
    Positions = np.array(Positions)
    print("The matched points are: \r\n",Positions)

    # 提取为4个list类型的值,便于计算基础矩阵
    px1 = Positions[:,0][:,0]
    py1 = Positions[:,0][:,1]
    px2 = Positions[:,1][:,0]
    py2 = Positions[:,1][:,1]

    P1 = np.row_stack((px1, py1, np.ones((1, 8), dtype=int)))
    P2 = np.row_stack((px2, py2, np.ones((1, 8), dtype=int)))  

    # 计算基础矩阵
    F = GetFdmtMatr(px1,py1,px2,py2)
    print("The fundmantal matrix F is: \r\n",F)

    print(F[2,2])
    StandardF = F / F[2,2]

    print('The StandardF matrix is:\r\n',StandardF)




    # F = F / F(2,2)

    # 根据计算原理验证计算精度
    result = P2.T @ F @ P1
    Error = []
    for i in range(8):
        Error.append(result[i][i])
    print("The Error is: \r\n", Error)

    # 计算本质矩阵
    E=np.dot(np.dot(K.T,F), K)
    print("The nature Matrix E is: ",E)

    R,T,whichRT = SolveRT(E,0,Positions)

    # matrix = [R[0,0],R[1,0];R[0,0],R[0,0]]
    # print(matrix)

    # TransMatrix = [
    #                 [R[0,0],R[0,1],R[0,2],T[0,2]],
    #                 [R[1,0],R[1,1],R[1,2],T[1,2]],
    #                 [R[2,0],R[2,1],R[2,2],T[2,2]],
    #                 [0,0,0,1]]

    # print(TransMatrix)

    # print(Positions[0,1])

    # Verify = Verify(Positions,R,T)
    # print(Verify)



    print("The rotate matrix R is: \r\n",R)
    print("The trans matrix T is: \r\n",T)
    print("The value of WhichRT is: \r\n",whichRT)

    # Verify(path1,path2,R,T)