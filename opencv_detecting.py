import cv2
import numpy as np
import math

# 得到蓝色的HSV指标
blue_lower = np.array([100, 43, 46])
blue_upper = np.array([124, 255, 255])

# 已知物体到摄像头的距离(cm)
KNOW_DISTANCE = 50

# 计算已知物体对应距离的实际宽(cm)
KONW_OBJECT_WIDTH = 7.48

# 计算已知物体对应距离的像素数目
KONW_OBJECT_PWIDTH = 67

# 获取摄像头对象
cap = cv2.VideoCapture(0)
# 利用相似三角形计算物体到摄像头的距离
def Get_Distance(W, F, P):
    '''
    :param W: 已知物体实际宽
    :param F: 相机的焦距
    :param P: 物体的像素宽
    :return:
    D_str: 返回字符串型的距离
    '''
    D = (W * F) / P
    # 将D 近似为两位小数
    D = round(D, 2)
    D_str = str(D)
    return  D_str


# 获取相机的焦距
def Get_LocalLonght(P, D ,W):
    '''
    :param P: 已知物体像素宽
    :param D: 相机位置也就是相机距离物体的距离
    :param W: 已知物体实际宽
    :return:
    F:相机的焦距
    '''
    F = (P * D) / W
    return  F

while(True):
    # 读取视频流
    flag, img = cap.read()

    # 转换为HSV颜色领域
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 制作掩膜，保存满足HSV指标的图像，其他过滤掉
    mask = cv2.inRange(img_HSV, lowerb= blue_lower, upperb= blue_upper)
    img_ball = cv2.bitwise_and(img, img, mask= mask)

    # 高斯模糊去噪
    img_ball = cv2.GaussianBlur(img_ball, (5,5), 0)
    cv2.imshow('img_ball', img_ball)

    # 转换为二值图
    img_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2GRAY)
    # 因为输入图像为灰度图所以阈值也就是亮度， 通过HSV指标过滤后背景还是会有些蓝色的干扰，所以设置一个合适的阈值将那些过亮的干扰过滤掉
    __, img_bin = cv2.threshold(img_ball, 120, 255, cv2.THRESH_BINARY)
    cv2.imshow('img_bin', img_bin)

    # 进行边缘检测，由于边缘检测阈值的限定，会对之后的轮廓检测省下一些操作
    img_canny = cv2.Canny(img_bin, 150, 250)
    cv2.imshow('img_canny', img_canny)

    # 之前试过形态学的一些操作，效果不如边缘检测
    # kernel = np.ones((9,9), dtype= np.uint8)
    # img_chli = cv2.erode(img_bin, kernel= kernel, iterations=6)
    # cv2.imshow('img_erod', img_chli)
    # img_chli = cv2.dilate(img_chli, kernel= kernel, iterations= 6)
    # cv2.imshow('img_dil', img_chli)

    # 轮廓检测
    contours, __ = cv2.findContours(img_canny,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 如果检测到的轮廓不会0
    if len(contours) != 0:
        # 寻找面积最大的轮廓，并获取最小外接矩形
        area = []
        for i in range(len(contours)):
            mianji = cv2.contourArea(contours[i])
            area.append(mianji)

        area = np.array(area, dtype= np.uint8)
        # 获取最大面积索引
        max_area_num = np.argmax(area)
        # 获取最大面积的轮廓
        cnt = contours[max_area_num]
        # 求它的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)

        # 画出最小外接矩形
        img_juxing = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)

        # 获取相机焦距
        F = Get_LocalLonght(KONW_OBJECT_PWIDTH, KNOW_DISTANCE, KONW_OBJECT_WIDTH)
        # 获取已经转换成字符串的距离
        D_STR = Get_Distance(KONW_OBJECT_WIDTH, F, w)

        # 打印距离
        img_juxing = cv2.putText(img, "distance:" + D_STR + "cm", (0,30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,255,0), bottomLeftOrigin= False)
        cv2.imshow('img_juxing', img_juxing)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
