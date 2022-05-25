import cv2
import mediapipe as mp

#module setup

class poseDetector():

	def __init__(self):
		self.mpPose =mp.solutions.pose
		self.mpDraw= mp.solutions.drawing_utils
		self.pose=self.mpPose.Pose()

	def findPose(self,img,draw=True):
		imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		results=self.pose.process(imgRGB)
		if results.pose_landmarks :
			if draw :
				self.mpDraw.draw_landmarks(img,results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,landmark_drawing_spec=self.mpDraw.DrawingSpec(color=(0,0,255),thickness=20, circle_radius=10),connection_drawing_spec=self.mpDraw.DrawingSpec(color=(0,255,0),thickness=20, circle_radius=20))
		return img

	def findPosition(self,img,lmlist,draw=True):
		imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		results=self.pose.process(imgRGB)
		if results.pose_landmarks :
			for id,lm in enumerate(results.pose_landmarks.landmark):
				if id in lmlist:
					h,w,c=img.shape
					cx,cy=int(lm.x*w),int(lm.y*h)
					if draw:
						cv2.circle(img,(cx,cy),10,(0,0,255),-1)
		return img

def main():
	
	#---------------------------------------write file name with its path instead of demo1.mp4
	#---------------------------------------or write = instaed of demo1.mp4 for webcam
	file="demo1.mp4"
	#file=0
	cap=cv2.VideoCapture(file)
	lmlist=[0,11,12,13,14,25,26,27,28]
	success, img=cap.read()
	detector=poseDetector()
	while success:	
		
		img1=cv2.resize(img, (500, 600),cv2.INTER_CUBIC)
		cv2.imshow("Original",img1)
		img=detector.findPose(img)
		#img=detector.findPosition(img,lmlist) 
		img=cv2.resize(img, (500, 600),cv2.INTER_CUBIC)
		cv2.imshow("Body Landmarks",img)
		success, img=cap.read()
		if success:
			key=cv2.waitKey(1)
		else :
			key=cv2.waitKey(0)
		if key!=-1 :
			break

	cv2.destroyAllWindows()

if __name__=="__main__":
	main()
