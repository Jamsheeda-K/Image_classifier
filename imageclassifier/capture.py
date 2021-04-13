import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam
from models.transferred_model import predict_class
from datetime import datetime



if __name__ == "__main__":

    # folder to write images to
    out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # 640x360
    # 640.0 x 480.0
    webcam = init_cam(640, 480)
    key = None
    predicted_category = 'Show me some things !!!'

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
            # get key event
            key = key_action()

            # draw a [224x224] rectangle into the frame, leave some space for the black border         
            cv2.rectangle(img=frame, pt1=(160-2+0,120-2+0), pt2=(160+2+224, 120+2+224), 
                          color=(0, 0, 0), thickness=2)  
            # font 
            font = cv2.FONT_HERSHEY_SIMPLEX 
  
            # org 
            org = (50, 50) 
  
            # fontScale 
            fontScale = 0.5
   
            # Blue color in BGR 
            color = (255, 0, 0) 
  
            # Line thickness of 2 px 
            thickness = 2
             # check if the image is in RGB
            
            frame = cv2.putText(frame, predicted_category, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
            image = frame[120+0:120+224, 160+0:160+224, :]
            predicted_category = predict_class(image)
            cv2.waitKey(500)

            if key == 'p':
                image = frame[120+0:120+224, 160+0:160+224, :]
                #write_image(out_folder, image) 
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #predicted_category = 'apple'
                # Using cv2.putText() method
                predicted_category = predict_class(image)

                #frame = cv2.putText(frame, predicted_category, org, font,  
                #    fontScale, color, thickness, cv2.LINE_AA) 
                #cv2.imshow("Frame", frame)
                #cv2.waitKey(100)
                #write_image(out_folder, image) 
                
                #predicted_category = predict_class(image)
                #predicted_category = 'apple'
    
                # Using cv2.putText() method 

            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[120+0:120+224, 160+0:160+224, :]
                write_image(out_folder, image) 

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()