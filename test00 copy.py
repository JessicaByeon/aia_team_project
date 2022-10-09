# import selenium 
# from selenium import webdriver 
# from selenium.webdriver.common.keys import Keys 
# driver = webdriver.Chrome("C:/chromedriver.exe")

# import os
# print(os.getcwd())

from selenium import webdriver
from selenium.webdriver.common.keys import Keys 
from selenium.webdriver.common.by import By
import time
import urllib.request

path = "C:/chromedriver.exe"
url = "https://www.google.co.kr/imghp?hl=ko&ogbl"

driver = webdriver.Chrome(path)
driver.implicitly_wait(5)
driver.get(url)

# 검색창을 찾는 것
elem = driver.find_element(By.XPATH, '//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img') # find_element 를 통해 원하는 종류를 선택하여 검색 가능
# 찾는 키워드 입력
elem.send_keys("이정재 얼굴")
# 엔터키 실행
elem.send_keys(Keys.RETURN)


# 이미지를 한꺼번에 모두 로딩시킨 후 진행하자.
# 기본적으로는 한 페이지에 50개가 업로드된 후 스크롤 다운을 계속적으로 하면서 추가 이미지가 로딩되는 방식으로 웹페이지가 구성
# 스크롤을 내려주는 코드

SCROLL_PAUSE_TIME = 1

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight") # 자바스크립트 코드 실행 --- 브라우저높이(스크롤높이) 체크 --- 값을 찾아서 last height에 저장

while True: # 무한반복
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # 브라우저 끝까지 스크롤을 내리겠다.

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height: # 스크롤 다운했을 때 더 나오는 것이 없다면 스크롤 높이가 같을 것 / 내렸을 때 추가 이미지가 있다면 스크롤이 길어진다면 new height > last height
        try:
            driver.find_element(By.CSS_SELECTOR, 'mye4qd').click()
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
images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
count = 1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_elements(By.XPATH, '//*[@id="Sva75c"]').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, str(count) + ".jpg") # 순차적으로 진행되면서 저장 --- 파일명 : 번호.jpg
        print("Image saved: 이정재_{}.jpg".format(count))
        count = count + 1
    except: # 오류발생 시 일단 무시하고 다음 작업 처리
        pass

driver.close() # 웹 브라우저를 닫아줌
