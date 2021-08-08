#使用k-means对图像进行聚类，显示分割标识的可视化
import numpy as np
import PIL.Image as image
from sklearn.cluster import KMeans
from sklearn import preprocessing
from skimage import color

#加载图像，并对数据进行可视化
def load_data(filePath):
    #读文件
    f = open(filePath,'rb')
    data = []
    img = image.open(f)
    width,height = img.size
    for x in range(width):
        for y in range(height):
            #得到点(x,y)的三个通道
            c1,c2,c3 = img.getpixel((x,y))
            data.append([c1,c2,c3])

    f.close()
    #采用Min-Max规范化
    mm =preprocessing.MinMaxScaler()
    data = mm.fit_transform(data)
    return np.mat(data),width,height

#加载图片，得到规范化的结果
img,widht,height = load_data('weixin.jpg')
#用K-means对图像进行16聚类
kmeans =KMeans(n_clusters=16)
kmeans.fit(img)
label =kmeans.predict(img)
#将图像聚类结果，转化成图像尺寸的矩阵
label =label.reshape([widht,height])
#将聚类标识矩阵转化为不同颜色的矩阵
label_color =(color.label2rgb(label)*255).astype(np.uint8)
label_color =label_color.transpose(1,0,2)
images =image.fromarray(label_color)
images.save('weixin-new.jpg')