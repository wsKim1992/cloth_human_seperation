import cv2
import numpy as np
import os
import shutil

def yolo(path,filename,frame,size,score_threshold,nms_threshold):
    # 클래스 리스트
    classes = ["person", "bicycle", "car", "motorcycle",
            "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
            "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
            "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    #클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0,255,size=(len(classes),3))

    #이미지의 높이, 너비, 채널 받아오기
    height,width, channels = frame.shape

    #네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame,1./255,(size,size),(0,0,0),False,crop=False)

    #전처리된 blob 네트워크에 입력
    net.setInput(blob)

    #결과 받아오기
    outs = net.forward(output_layers)

    #각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                #탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x-w / 2)
                y = int(center_y-h /2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #후보 박스와 confidence(상자가 물체일 확률) 출력
    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    #Non Maximum Supppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,score_threshold=score_threshold,nms_threshold=nms_threshold)

    #후보 박스 중 선택된 박스의 인덱스 출력
    print(f"indexes: ",end='')
    for index in indexes:
        print(index,end=' ')
    print("\n\n============================ classes ============================")
    
    is_people_exist = False
    max_confi_val = 0
    max_class_id = -1
    max_idx = -1
    peop_idx = -1
    for i in range(len(boxes)):
        if i in indexes:
            if class_ids[i]==0:
                is_people_exist=True
                peop_idx=i
            if confidences[i]>max_confi_val:
                max_confi_val=confidences[i]
                max_class_id=class_ids[i]        
                max_idx = i
            x,y,w,h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            
            #탐지된 객체의 정보 출력
            print(f"[{class_name}({i})] conf: {confidences[i]} / x:{x} \ y:{y} \ width:{w} / height: {h}")
    flag = False
    if max_class_id == -1 or is_people_exist ==False:
        flag = False
    elif max_class_id==0: 
        flag = True
    else :
        x1,y1,w1,h1=boxes[max_idx]
        x2,y2,w2,h2=boxes[peop_idx]
        max_area = w1*h1
        peop_area = w2*h2
        if peop_area>max_area and (max_area/peop_area)>0.3:
           flag=True
    
    if flag==True:
        new_filename='people_'+filename
        print(new_filename)
        os.rename(os.path.join(path,filename),os.path.join(path,new_filename))

    return frame

def do_yolo(root_path):
    dirs = os.listdir(root_path)

    for dir in dirs:
        dir_path = os.path.join(root_path,dir)
        files = os.listdir(dir_path)
        for file in files :
            filename,extension = os.path.splitext(os.path.join(dir_path,file))
            prefix = file.split('_')[0]
            print('filename : '+filename)
            print('prefix : '+prefix)
            if prefix !='etc' and (extension=='.jpg' or extension == '.png' or extension == '.jpeg'):
                full_path = os.path.join(dir_path,file)
                frame = cv2.imread(full_path)
                size_list=[320,416,608]
                print(full_path)
                frame = yolo(path=dir_path,filename=file,frame=frame, size = size_list[2],score_threshold=0.4,nms_threshold=0.4)
                
                """ cv2.imshow("yolo_output",frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows() """
