import matplotlib.pyplot as plt
import cv2
from FastSAM_inference import get_fSAM
import cv2 

def main(): 
	object = get_fSAM()
	img = object.infer('resources/images/dog.jpg')
	vid = cv2.VideoCapture('resources/video/0.mp4') 
	h= 640
	w = 480
	print(type(img))
	cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
	cv2.imshow('Image', img) 
	cv2.waitKey(1)

	while(True): 
		
		ret, frame = vid.read() 

		if ret == True:
			cv2.imshow('frame', frame)
		
			if cv2.waitKey(1) & 0xFF == ord('q'): 
				break
		else:
			break

	vid.release() 
	cv2.destroyAllWindows() 

if __name__ == "__main__":
	main()
