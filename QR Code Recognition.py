import numpy as np
import copy
import math
import cv2
import os


def pre_process(img, flag=0):
    # 调整尺寸
    width, height = img.shape[1], img.shape[0]
    if flag == 0:
        img = cv2.resize(img, (400, int(height / (width / 400))))
    else:
        img = cv2.resize(img, (500, int(height / (width / 500))))
    # 高斯滤波
    blurred = cv2.GaussianBlur(img, (3, 3), 0.1)
    if flag == 1:
        # 直方图归一化
        blurred = cv2.equalizeHist(blurred)
    # 自适应二值化
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 12)
    # 轮廓提取
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return img, contours, hierarchy


def judge_contours(contours, i, j, k):
    # 计算二维码定位块轮廓面积比例是否合适
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    area3 = cv2.contourArea(contours[k])
    if area2 == 0 or area3 == 0:
        return False
    ratio1 = float(area1) / area2
    ratio2 = float(area2) / area3
    if abs(ratio1 - 49.0 / 25) < 2 and abs(ratio2 - 25.0 / 9) < 2:
        return True
    return False


def contours_center(contours, i):
    # 计算轮廓中心点
    M = cv2.moments(contours[i])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


def judge_centers(ct):
    # 判断三个轮廓中心是否接近
    distance_1 = np.sqrt((ct[0] - ct[2]) ** 2 + (ct[1] - ct[3]) ** 2)
    distance_2 = np.sqrt((ct[0] - ct[4]) ** 2 + (ct[1] - ct[5]) ** 2)
    distance_3 = np.sqrt((ct[2] - ct[4]) ** 2 + (ct[3] - ct[5]) ** 2)
    if sum((distance_1, distance_2, distance_3)) / 3 < 3:
        return True
    return False


def judge_triangle(ct):
    # 判断是否能构成等腰直角三角形
    if len(ct) < 3:
        return -1, -1, -1
    for i in range(len(ct)):
        for j in range(i + 1, len(ct)):
            for k in range(j + 1, len(ct)):
                distance_1 = np.sqrt((ct[i][0] - ct[j][0]) ** 2 + (ct[i][1] - ct[j][1]) ** 2)
                distance_2 = np.sqrt((ct[i][0] - ct[k][0]) ** 2 + (ct[i][1] - ct[k][1]) ** 2)
                distance_3 = np.sqrt((ct[j][0] - ct[k][0]) ** 2 + (ct[j][1] - ct[k][1]) ** 2)
                thresh = 20
                # 判断中心点是否重合
                if abs(distance_1) <= 5 or abs(distance_2) <= 5 or abs(distance_3) <= 5:
                    continue
                # 判断是否是等腰直角三角形
                if abs(distance_1 - distance_2) < thresh:
                    if abs(np.sqrt(np.square(distance_1) + np.square(distance_2)) - distance_3) < thresh:
                        return i, j, k
                if abs(distance_1 - distance_3) < thresh:
                    if abs(np.sqrt(np.square(distance_1) + np.square(distance_3)) - distance_2) < thresh:
                        return j, i, k
                if abs(distance_2 - distance_3) < thresh:
                    if abs(np.sqrt(np.square(distance_2) + np.square(distance_3)) - distance_1) < thresh:
                        return k, i, j
    return -1, -1, -1


def QR_detector(image, contours, hierachy, root=0):
    # 寻找轮廓
    con = []
    for i in range(len(hierachy)):
        j = hierachy[i][2]
        k = hierachy[j][2]
        if j != -1 and k != -1:
            if judge_contours(contours, i, j, k):
                cx1, cy1 = contours_center(contours, i)
                cx2, cy2 = contours_center(contours, j)
                cx3, cy3 = contours_center(contours, k)
                if judge_centers([cx1, cy1, cx2, cy2, cx3, cy3]):
                    con.append([cx1, cy1, cx2, cy2, cx3, cy3, i, j, k])

    # 判断识别定位块
    i, j, k = judge_triangle(con)
    for point in con:
        p = (point[0], point[1])
        cv2.circle(image, p, 3, (255, 0, 0), 3)
    if not (i == -1 or j == -1 or k == -1):
        # 寻找最小外接矩形
        ts = np.concatenate((contours[con[i][6]], contours[con[j][6]], contours[con[k][6]]))
        rect = cv2.minAreaRect(ts)
        box = np.int0(cv2.boxPoints(rect))

        # cv2.drawContours(image, contours, rec[i][6], (255, 0, 0), 2)
        # cv2.drawContours(image, contours, rec[j][6], (255, 0, 0), 2)
        # cv2.drawContours(image, contours, rec[k][6], (255, 0, 0), 2)
        # cv2.imshow('img', image)
        # cv2.waitKey(0)

        return box
    else:
        return [[0]]


def process(img_list, result_list, flag):
    cnt = 0
    for i in img_list:
        img, contours, hierachy = pre_process(i)
        # tmp = cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
        # cv2.imshow('contours', tmp)
        # cv2.waitKey(0)
        box = QR_detector(img, contours, np.squeeze(hierachy))
        if box[0][0] != 0:
            result = result_list[cnt]
            width, height = result.shape[1], result.shape[0]
            result = cv2.resize(result, (400, int(height / (width / 400))))
            cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
            # cv2.imshow('img', result)
            # cv2.waitKey(0)
            if flag == 1:
                cv2.imwrite('Result/easy/%d.jpg' % (cnt+1), result)
            else:
                cv2.imwrite('Result/hard/%d.jpg' % (cnt + 1), result)
            cnt += 1
        else:
            img, contours, hierachy = pre_process(i, flag=1)
            box = QR_detector(img, contours, np.squeeze(hierachy))
            if box[0][0] != 0:
                result = result_list[cnt]
                width, height = result.shape[1], result.shape[0]
                result = cv2.resize(result, (500, int(height / (width / 500))))
                cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
                # cv2.imshow('img', result)
                # cv2.waitKey(0)
                if flag == 1:
                    cv2.imwrite('Result/easy/%d.jpg' % (cnt + 1), result)
                else:
                    cv2.imwrite('Result/hard/%d.jpg' % (cnt + 1), result)
                cnt += 1
            else:
                cnt += 1
                print('Not Find')


if __name__ == '__main__':
    # Read Images
    easy_img = []
    hard_img = []
    result_easy = []
    result_hard = []
    for file in os.listdir('easy'):
        file_name = os.path.join('easy', file)
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        easy_img.append(gray)
        result_easy.append(img)
    for file in os.listdir('hard'):
        file_name = os.path.join('hard', file)
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hard_img.append(gray)
        result_hard.append(img)

    process(easy_img, result_easy, 1)

    process(hard_img, result_hard, 2)
