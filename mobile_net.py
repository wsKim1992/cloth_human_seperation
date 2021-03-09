import numpy as np 
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot
import os
import shutil
import cv2 as cv2
import json
import sys

def classify(arg):
    #'./data/people',
    data,sample = arg.split('/')
    path = os.path.join(data,sample)
    print(path)
    dirs = os.listdir(path)
    num_test=4
    classified_list={}
    model = MobileNet(weights='imagenet')
    if os.path.exists(os.path.join('./cloth_label.txt')):
        f = open('./cloth_label.txt','r')
        categories = f.read().split()
        f.close()
    else :
        return False
    print(categories)
    for category in categories:
        classified_list[category]=0

    for dir in dirs :
        dir_path = os.path.join(data,sample,dir)
        files = os.listdir(dir_path)
        for file in files:
            name,extension = os.path.splitext(os.path.join(dir_path,file))
            if extension=='.jpg' or extension=='.png' or extension=='.jpeg':     
                src = os.path.join(dir_path,file)
                print("src : "+src)
                img = image.load_img(src,target_size=(224,224))
                x = image.img_to_array(img)
                x = np.expand_dims(x,axis=0)
                x=preprocess_input(x)
                preds = model.predict(x)
                labels = decode_predictions(preds,top=num_test)[0]
                flag =False
                
                for label in labels:
                    label = list(label)
                    #print(classified_list.get(label[1]))
                    if classified_list.get(label[1])!=None:
                        """ save_dir='./cloth_and_people'
                        shutil.move(src,dst) """
                        """ if not(os.path.isdir(os.path.join(save_dir))):
                            os.makedirs(os.path.join(save_dir))
                        dst = os.path.join(save_dir,file) """
                        flag=True
                        break
                if not(flag):
                    label = 'etc'
                    new_filename = 'etc_'+file
                    print(new_filename)
                    os.rename(os.path.join(dir_path,file),os.path.join(dir_path,new_filename))
                    """ save_dir='./etc' """
                    """ if dir =='./data/cloth_and_people':
                        save_dir = 'etc_'+str(num_test)
                    else:
                        save_dir='etc_'+str(num_test)+'_etc' """
                    """ if not(os.path.exists(save_dir)):
                        os.makedirs(os.path.join(save_dir))
                    shutil.move(src,os.path.join(save_dir,file)) """ 
        
if __name__ == "__main__":
    classify(arg)