import cv2
import math
import os
import json

def output_keypoints(image_path,filename,proto_file,weights_file,threshold,model_name,BODY_PARTS):
    global points
    # 이미지 읽어오기
    frame = cv2.imread(os.path.join(image_path,filename))

    # 원본 이미지의 높이, 너비를 받아오기
    frame_height, frame_width = frame.shape[:2]

    #네트워크 불러오기
    net = cv2.dnn.readNetFromCaffe(proto_file,weights_file)
    #입력 이미지의 사이즈 정의
    image_height = int(frame_height/10)
    image_width = int(frame_width/10)

    input_blob = cv2.dnn.blobFromImage(frame,1.0/255,(image_width,image_height),(0,0,0),swapRB=False,crop=False)
    #전처리된 blob 네트워크에 입력
    net.setInput(input_blob)

    #결과 받아오기 
    out = net.forward()
    # The output is a 4D matrix :
    # The first dimension being the image ID ( in case you pass more than one image to the network ).
    # The second dimension indicates the index of a keypoint.
    # The model produces Confidence Maps and Part Affinity maps which are all concatenated.
    # For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points.
    # We will be using only the first few points which correspond to Keypoints.
    # The third dimension is the height of the output map.
    out_height = out.shape[2]
    # The fourth dimension is the width of the output map.
    out_width = out.shape[3]

    

    #point 리스트 초기화
    points=[]

    flag = False

    for i in range(len(BODY_PARTS)):
        #신체부위의 CONFIDENCE MAP
        prob_map = out[0,i,:,:]
        #최소값 , 최대값, 최소값 위치, 최대값 위치
        min_val,prob,min_loc,point= cv2.minMaxLoc(prob_map)

        #원본 이미지에 맞게 포인트 위치 조정
        x = (frame_width*point[0])/out_width
        x = int(x)
        y = (frame_height*point[1])/out_height
        y = int(y)

        #pointed
        if prob > threshold:
            cv2.circle(frame,(x,y),5,(0,255,255),thickness=-1,lineType=cv2.FILLED)
            cv2.putText(frame,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),1,lineType=cv2.LINE_AA)
            points.append((x,y))
            if i!=25:
                flag = True
        else : # not pointed
            cv2.circle(frame,(x,y),5,(0,255,255),thickness=-1,lineType=cv2.FILLED)
            cv2.putText(frame,str(i),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),1,lineType=cv2.LINE_AA)
            points.append(None)
    print(flag)
    if flag==False:
        token = filename[:filename.find('_')+1]
        print(f"token : {token}")
        new_filename = filename.replace(token,'',1)
        os.rename(os.path.join(image_path,filename),os.path.join(image_path,new_filename))



    return frame

def output_keypoints_with_lines(POSE_PAIRS,BODY_PARTS,frame,filename):
    global not_linked_dic
    frame_line = frame.copy()

    if (points[1] is not None) and (points[8] is not None):
        calculate_degree(point_1=points[1],point_2=points[8],frame=frame_line)
    
    for pair in POSE_PAIRS:
        part_a=pair[0]
        part_b=pair[1]
        if points[part_a] and points[part_b]:
            if part_a == 1 and part_b ==8:
                cv2.line(frame,points[part_a],points[part_b],(255,0,255),3)
            else :
                cv2.line(frame,points[part_a],points[part_b],(0,255,0),3) 
        else:
            print(f"[not linked] {part_a} {points[part_a]} <==> {part_b} {points[part_b]}")
            not_linked_dic[filename]=(BODY_PARTS[part_a],BODY_PARTS[part_b])
    frame_horizontal = cv2.hconcat([frame,frame_line])
    """ if not(os.path.isdir(os.path.join('./people_with_frame',sub_path))):
        os.makedirs(os.path.join('./people_with_frame',sub_path)) """
    cv2.imwrite(os.path.join('./people_with_frame',f'{filename}.jpg'),frame_horizontal)
    #cv2.imshow("Output Keypoints With Lines",frame_horizontal)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def calculate_degree(point_1,point_2,frame):
    #역탄젠트 구하기
    dx = point_2[0]-point_1[0]
    dy = point_2[1]-point_1[1]
    rad = math.atan2(abs(dy),abs(dx))

    deg = rad*180 /math.pi

    if deg<45:
        string="Bend Down"
        cv2.putText(frame,string,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
    else:
        string="Stand"
        cv2.putText(frame,string,(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))

# MPII 에서 각 파트 번호, 선으로 연결될 POST_PAIRS
BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

POSE_PAIRS_BODY_25 = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]

#키포인트를 저장할 빈 리스트
points = []
not_linked_dic={}

def do_pose_detection(root_dir):
    dirs = os.listdir(root_dir)
    #신경 네트워크의 구조를 지정하는 prototxt 파일 (다양한 계층이 배열되는 방법 등)
    protoFile_body_25 = "C:\\Users\\G\\Documents\\GitHub\\openpose\\models\\pose\\body_25\\pose_deploy.prototxt"
    #훈련된 모델의 weight 를 저장하는 caffemodel 파일
    weightFile_body_25 = "C:\\Users\\G\\Documents\\GitHub\\openpose\\models\\pose\\body_25\\pose_iter_584000.caffemodel"
    
    for dir in dirs:
        dir_path = os.path.join(root_dir,dir)
        print(dir_path)
        files=os.listdir(dir_path)
        for file in files:
            filename,extension = os.path.splitext(file)
            prefix = filename.split("_")[0]
            """ print(prefix)
            print(extension) """
            if prefix=="people" and (extension==".jpg" or extension==".png" or extension==".jpeg"):
                frame = output_keypoints(image_path=dir_path,filename=file,proto_file=protoFile_body_25,weights_file=weightFile_body_25,threshold=0.1,model_name=file,BODY_PARTS=BODY_PARTS_BODY_25)
                output_keypoints_with_lines(POSE_PAIRS=POSE_PAIRS_BODY_25,BODY_PARTS=BODY_PARTS_BODY_25,frame=frame,filename=file)    
            




