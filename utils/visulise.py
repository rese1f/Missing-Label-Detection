from pycocotools.coco import COCO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize(dataDir, dataType, annFile):
    dataDir='/path/to/your/coco_data'
    dataType='val2017'
    annFile=annFile.format(dataDir,dataType)      #'{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco=COCO(annFile)
    
    
    





def plot_picture(coco):
    # get all images containing given categories, select one at random
    catIds = coco.getCatIds(catNms=['person','dog','skateboard']);
    imgIds = coco.getImgIds(catIds=catIds );
    imgIds = coco.getImgIds(imgIds = [324158])
    # loadImgs() 返回的是只有一个元素的列表, 使用[0]来访问这个元素
    # 列表中的这个元素又是字典类型, 关键字有: ["license", "file_name", 
    #  "coco_url", "height", "width", "date_captured", "id"]
    img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

    # 加载并显示图片,可以使用两种方式: 1) 加载本地图片, 2) 在线加载远程图片
    # 1) 使用本地路径, 对应关键字 "file_name"
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))  

    # 2) 使用 url, 对应关键字 "coco_url"
    I = io.imread(img['coco_url'])        
    plt.axis('off')
    plt.imshow(I)
    plt.show()



