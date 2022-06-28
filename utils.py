import cv2
from cv2 import boundingRect

def sort_contours(cnts, method="left-to-fight"):
  """ 排序函数 """
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    
    boundingBoxes = [cv2.boundingRect(c) for c in cnts] # 用一个最小的矩形把找到的形状包起来
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

def cv_show(name, img):
  """ 单张图展示 """
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hstack_show(*img):
  """ 多张图展示 """
    res = np.hstack((img))
    return res
