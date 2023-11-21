import matplotlib.pyplot as plt
import cv2
from FastSAM_inference import get_fSAM
import cv2 

def main(): 
	object = get_fSAM()

	vid = cv2.VideoCapture('resources/video/0.mp4') 
	

	while(True): 
		
		ret, frame = vid.read() 

		if ret == True:
			seg = object.infer(frame, frame.shape[1]//2)
			cv2.imshow('segmented image', seg)
			if cv2.waitKey(1) & 0xFF == ord('q'): 
				break
		else:
			break

	vid.release() 
	cv2.destroyAllWindows() 

if __name__ == "__main__":
	main()
