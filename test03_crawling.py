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

if not os.path.isdir("이정재/"):
    os.makedirs("이정재/")

# driver = webdriver.Chrome('C://chromedriver.exe')
driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")

keywords = "이정재 얼굴"
elem = driver.find_element(By.XPATH,"/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input")

elem.send_keys(keywords)
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE_TIME)
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR,".mye4qd").click()
        except:
            break
    last_height = new_height

images = driver.find_elements(By.CSS_SELECTOR,".rg_i.Q4LuWd")
count = 1

for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = driver.find_element(By.XPATH,"//*[@id='Sva75c']/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[3]/div/a/img").get_attribute('src')
        urllib.request.urlretrieve(imgUrl, "이정재/" + keywords + "_" + str(count) + ".jpg")
        print("Image saved: 이정재_{}.jpg".format(count))
        count += 1
    except:
        pass

driver.close()