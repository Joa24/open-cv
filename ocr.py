# -*- coding: utf-8 -*-

from string import digits
import numpy as np
import argparse
import cv2
from utils.myutils import sort_contours

#设置参数：原图像和模板图像
ap = argparse.ArgumentParser()
ap.add_argument("--images", default="./images/card_04.jpg", help="path to input image")
ap.add_argument("--template", default="./images/card_template.jpg", help="path tot template OCR-A image")
args = vars(ap.parse_args())

#绘图展示函数
def cv_show(name,img):
  cv2.imshow(name,img)
  cv2.waitKey()
  cv2.destroyAllWindows()
  
############################################ 1.读取模版图像并预处理 ############################################
temp = cv2.imread(args['template'])

#模版图像变为灰度图
ref = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

#灰度图变为二值图
#cv2.THRESH_BINARY_INV:设定阈值，大于则为0，小于则为255；返回值为阈值和结果图像，取结果图像即可
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

#计算轮廓
#cv2.findContours:用于查找图像轮廓，参数1二值图，参数2轮廓的检索模式，参数3轮廓的近似办法；返回值1为轮廓本身，返回值2为每条轮廓对应的属性
ref_, refCnts, hierarchy = cv2.findContours(ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#绘制轮廓信息
#在模版图像上绘制轮廓，-1表示每个图形都要绘制，(0,0,255)为红色，线条粗细为3
cv2.drawContours(temp,refCnts,-1,(0,0,255),3)

# np.array(refCnts).shape 可以验证数量是否为10

#由于选出的数字轮廓特征不一定按照顺序从0-9排序，为了和模板的数字顺序匹配，自定义一个排序函数
#得到的返回值为排序后的属性值
refCnts = sort_contours(refCnts, method='left-to-right')[0]
digits = {}

#通过遍历每个属性，并绘制边界矩形，获取4值值
for (i,c) in enumerate(refCnts):
  (x,y,w,h) = cv2.boundingRect(c)
  roi = ref[y:y+h,x:x+w]
  roi = cv2.resize(roi, (57,88))
  
#初始化之后需要用到的卷积核，卷积核的大小根据图像特征自定义
rectkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

############################################ 2.读取原始图像并预处理 ############################################
image = cv2.imread(args['images'])
image = cv2.resize(image, (300, 190))
gray = cv3.cvtColor(image, cv2.COLOR_BGR2GRAY)

#要获取信用卡中颜色较为突出的部分(白色数字)，可以对图像先做腐蚀，模糊白色的部分，此时信用卡主体为黑色，在用原图像 - 操作完的图像，即可保留亮色的部分
#也可以直接使用礼帽操作(原始图-开运算(先腐蚀后膨胀))
result_img = gray - cv2.erode(gray, rectkernel, iterations=2)

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHT, rectkernel)

#计算梯度，获取边缘信息
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=1)
gradX = np.absolute(gradX)
#获取最大最小值后对图像进行归一化，然后用归一化后的数值*255，得到边缘的线条
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype('uint8')

#得到边缘信息后，需要将4组卡号数字组合，因此用膨胀可以将每4个数字组合,在用腐蚀操作去掉一些可能误连接在一起的细微特征(或单用腐蚀操作)
#闭运算=膨胀-腐蚀
gradX = cv2.morplologyEx(gradX, cv2.MORPH_CLOSE, rectkernel)

#cv2.THRESH_OTSU方法可以自动选择合适的阈值，选择出的值可以作为阈值参数传入cv2.threshold函数中
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#为了填充分组后的空缺，再进行一次闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

#计算轮廓
thresh, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cur_img = cv2.drawContours(image.copy(), threshCnts, -1, (0,0,255), 3)

#遍历轮廓信息，选处代表卡号的4组轮廓；制定的选择策略为：选择y坐标相同的4个点
locs, li_y = [], []
for (i, c) in enumerate(cnts):
    (_,y,_,_) = cv2.boundingRect(c)

    li_y.append(y)

res_y = max(set(li_y),key=li_y.count)

for (i,c) in enumerate(cnts):
    (x,y,w,h) = cv2.boundingRect(c)
    if y == res_y:
        locs.append((x,y,w,h))
        
#将选择出的轮廓进行排序,按照x的坐标，从左到右
locs = sorted(locs, key=lambda x: x[0])

#遍历4组轮廓中的每一个数字
groupOutput = []
for (i,(gx,gy,gw,gh)) in enumerate(locs):
  groupOutput = []
  #根据坐标提取每组，参数5为自定义
  group = grad[gy-5 : gy+gh+5, gx-5: gx+gw+5]
  group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

  #计算每一组的轮廓
  group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  digitCnts = sort_contours(digitCnts,method='left-to-right')[0] # 自定义排序函数

  # 计算每一组中的每一个数值
  for c in digitCnts:
      (x,y,w,h) = cv2.boundingRect(c)
      roi = group[y:y+h, x:x+w]
      roi = cv2.resize(roi, (57,88))
      # cv_show('roi', roi)

      scores = []
      
      #采用模版匹配方法，匹配原图像中的每个数字；matchTemplate函数返回值为相关系数矩阵，因此采用minMaxLoc方法，获取最大相关系数值
      for(digit, digitROI) in digits.items():
        result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
        (_, score, _, _) = cv2.minMaxLoc(result)
        socres.append(score)
      #然后采用numpy的argmax函数取出所有score中最大值的索引
      groupOutput.append(str(np.argmax(scores)))

  #绘制边界矩形
  cv2.rectangle(image, (gx-5,gy-5),(gx+gw+5),gy+gh+5),(0,0,255),1)
  #打印数字在原图像上
  cv2.putText(image, "".join(groupOutput), (gx,gy-15),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
  #
  output.extend(groupOutput)
  
print('credit card #:{}'.format("".join(output)))
cv2.imshow("image",image)
cv2.waitKey()
cv2.destroyAllWindows() 
