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

    if new_height == last_height:  # 스크롤 다운했을 때 더 나오는 것이 없다면 스크롤 높이가 같을 것 / 내렸을 때 추가 이미지가 있다면 스크롤이 길어진다면 new height > last height
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


