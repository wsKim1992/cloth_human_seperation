import os
import sys
from mobile_net import classify 
from yolo import do_yolo
from pose_detection import do_pose_detection
def main(argv):
    print(argv[1])
    data_path=argv[1]
    if os.path.isdir(os.path.join(data_path)):
        classify(data_path)
        do_yolo(data_path)            
        #do_pose_detection(data_path)

if __name__=="__main__":
    main(sys.argv)