#import numpy as np 
#from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
#from tensorflow.keras.preprocessing import image
#import matplotlib.pyplot
import os
import shutil

def classify():
    #'./data/people',
    dirs = [
        'data/cloth_and_people',
    ]
    
    classified_list={}
    #model = MobileNet(weights='imagenet')
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
        idx=0
        files = os.listdir(dir)
        for file in files:
            src = os.path.join(dir,file)
            #img = image.load_img(src,target_size=(224,224))
            #x= image.img_to_array(img)
            #x = np.expand_dims(x,axis=0)
            #x=preprocess_input(x)
            #preds = model.predict(x)
            #labels = decode_predictions(preds,top=3)[0]
            flag =False
            dst = os.path.join('output','cloth_and_people',file)
            shutil.move(src,dst)
            """ for label in labels:
                label = list(label)
                print(label[1])
                print(classified_list.get(label[1]))
                if classified_list.get(label[1])!=None:
                    classified_list[label[1]]+=1
                    if not(os.path.isdir(os.path.join('data','cloth_and_people'))):
                        os.makedirs(os.path.join('data','cloth_and_people'))
                    dst = os.path.join('data','cloth_and_people')
                    f= open(os.path.join(dst,file),'wb')
                    f.close()
                    flag=True
                    break
            if not(flag):
                if not(os.path.exists(os.path.join('./data','etc'))):
                    os.makedirs(os.path.join('./data','etc'))
                shutil.move(full_filename,os.path.join('./data','etc',file))
             """    
if __name__ == "__main__":
    classify()