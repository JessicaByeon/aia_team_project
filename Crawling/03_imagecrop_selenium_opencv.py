from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen
import time
import urllib.request
import os

# options = webdriver.ChromeOptions()
# options.add_argument('headless')
# options.add_argument('window-size=1920x1080')
# options.add_argument('disable-gpu')
# options.add_argument('User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.95 Safari/537.36')
# options.add_argument('window-size=1920x1080')
# options.add_argument('ignore-certificate-errors')
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_experimental_option("excludeSwitches",["enable-logging"])
driver = webdriver.Chrome('C://chromedriver.exe',chrome_options=options)

# 지정한 폴더가 없을경우, 지정한 폴더의 이름으로 된 폴더를 새롭게 생성
if not os.path.isdir("D:\study_data\_image/이정재2/"):
    os.makedirs("D:\study_data\_image/이정재2/")

# driver = webdriver.Chrome('C://chromedriver.exe')
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

# 검색창을 찾고, 찾을 키워드를 정한 후, 엔터키 실행
keywords = "이정재 얼굴"
elem = driver.find_element(By.XPATH,"/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")

elem.send_keys(keywords)
elem.send_keys(Keys.RETURN)


# 이미지를 한꺼번에 모두 로딩시킨 후 진행하자.
# 기본적으로는 한 페이지에 50개가 업로드된 후 스크롤 다운을 계속적으로 하면서 추가 이미지가 로딩되는 방식으로 웹페이지가 구성
# 스크롤을 내려주는 코드

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_heigh
ㅔt:  # 스크롤 다운했을 때 더 나오는 것이 없다면 스크롤 높이가 같을 것 / 내렸을 때 추가 이미지가 있다면 스크롤이 길어진다면 new height > last height
        try:
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
        except: # 결과 더보기 버튼을 추가적으로 누른 후에도 결과값이 더 이상 없을 경우
            break
    last_height = new_height


# #####################################################################################
# # 작은 이미지 선택
# elem = driver.find_elements(By.CSS_SELECTOR, "#islrg > div.islrc > div.isv-r.PNCib.MSM1fd.BUooTd.fT6ABc > a.wXeWr.islib.nfEiy > div.bRMDJf.islir > img")[0].click() # "class 입력", 가장 첫번째 이미지를 선택 [0], 클릭하겠다 .click()
# time.sleep(3)
# # 큰 이미지 선택 -- 큰 이미지 태그 찾고, src 주소를 가져옴
# imgUrl = driver.find_elements_by_css_selector("").get_attribute("src")
# # 이미지 다운로드
# urllib.request.urlretrieve(imgUrl, ".jpg") # 이미지 url, 저장하고자 하는 이름
# #####################################################################################


##### 여러개의 이미지를 한꺼번에 저장하는 for문 #####
# 작은 이미지 확인 -> 클릭 -> 큰 이미지 
images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
count = 1

print('==========')

for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element(By.XPATH,"//*[@id='Sva75c']/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img").get_attribute('src')
        urllib.request.urlretrieve(imgUrl, "D:\study_data\_image/이정재2/" + keywords + "_" + str(count) + ".jpg") # 순차적으로 진행되면서 저장 --- 파일명 : 번호.jpg
        print("Image saved: 이정재_{}.jpg".format(count))
        count += 1
    except: # 오류발생 시 일단 무시하고 다음 작업 처리
        pass

driver.close() # 웹 브라우저를 닫아줌



import cv2
import numpy as np
import os

path_dir = "D:\study_data\_image/이정재2/"
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
    # detectMultiScale 함수로 얼굴로 판단되는 부분의 정보를 이미지에서 검출
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
        cv2.imwrite(f"D:\study_data\_image/이정재2/{name}.jpg", resize)

# 지정한 경로의 모든 사진들에서 이미지를 검출해 자르고 저장하는 작업 수행 코드     
for name in file_name_list:
    img = cv2.imread("D:\study_data\_image/이정재2/"+name+".jpg")
    Cutting_face_save(img, name)
    
print('Done')








