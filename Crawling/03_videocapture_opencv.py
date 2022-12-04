import cv2

# OpenCV에서는 카메라와 동영상으로부터 프레임(frame)을 받아오는 작업을 
# cv2.VideoCapture 클래스 하나로 처리
# 카메라/동영상 열기
vidcap = cv2.VideoCapture("D:\study_data/1.mp4")
count = 0

print('=====')

# 비디오 캡쳐가 준비되었는지 확인
while(vidcap.isOpened()):
    ret, image = vidcap.read() # 프레임 받아오기
    cv2.imwrite("D:\study_data/frame%d.jpg" % count,image)
    print("Saved frame%d.jpg" % count)
    count += 1
    
vidcap.release()

################## 1/20 프레임 단위로 저장
'''   
while(vidcap.isOpened()):                       
    ret, image = vidcap.read()
    
    if(int(vidcap.get(1))% 20 == 0):
        cv2.imwrite("D:\study_data/_video/video_01/frame%d.jpg" % count,image)
        print("Saved fram%d.jpg" % count)
        count += 1
'''
########################################


import cv2
import numpy as np
import os

path_dir = "D:\study_data\_video/video_01/"
file_list = os.listdir(path_dir)

# print(file_list[0])
# print(len(file_list))

file_name_list = []

for i in range(len(file_list)):
    file_name_list.append(file_list[i].replace(".jpg","")) # 최종 이미지 저장 시 이름이 '이름.jpg.jpg'가 되는 것을 방지
print(file_name_list)


# 여러장의 사진을 작업하기 위한 코드 함수화
# print(file_name_list[0])
def Cutting_face_save(image, name):
    # 얼굴을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # 얼굴 검출할 그레이스케일 이미지 준비
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 이미지에서 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 얼굴 위치에 대한 좌표 정보를 리턴
    for (x,y,w,h) in faces:
        # 원본 이미지에 얼굴 위치 표시
        # cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
        cropped = img[y - int(h/3):y + h + int(h/3), x - int(w/3):x + w + int(w/3)] # cropped = image[y: y+h, x: x+w]
        resize = cv2.resize(cropped, (512,512))
        
        # cv2.imshow("crop&resize", resize)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 이미지 저장하기
        cv2.imwrite(f"D:\study_data\_video/video_01/{name}.jpg", resize)

# 지정한 경로의 모든 사진들에서 이미지를 검출해 자르고 저장하는 작업 수행 코드     
for name in file_name_list:
    img = cv2.imread("D:\study_data\_video/video_01/"+name+".jpg")
    Cutting_face_save(img, name)
    
print('Done')










