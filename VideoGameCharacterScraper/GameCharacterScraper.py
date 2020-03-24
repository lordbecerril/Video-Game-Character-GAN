'''
 AUTHORS: (in no particular order)
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>

 PURPOSE:
        The purpose of this script is to create a dataset from scraping
        video game characters from a video game character database. We
        utlized python selenium and the data was stored in ./data directory... enjoy :)
'''
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from io import StringIO
from datetime import datetime
import pandas as pd
import requests
import time
import urllib
import urllib.request
from string import ascii_lowercase



def main():
    print("Hello World, From The Video Game Character Web Scraper!")
    options = Options()
    options.add_argument("no-sandbox")
    options.add_argument("headless")
    options.add_argument('--no-proxy-server')
    options.add_argument("start-maximized")
    options.add_argument("window-size=1900,1080");
    driver = webdriver.Chrome(options=options, executable_path="chromedriver")
    driver.set_page_load_timeout(15)
    count = 0
    pages = ['http://www.game-art-hq.com/52008/the-video-game-character-database-letter-a/','http://www.game-art-hq.com/52158/video-game-characters-on-game-art-hq-b/','http://www.game-art-hq.com/52659/video-game-characters-on-game-art-hq-c/','http://www.game-art-hq.com/52545/video-game-characters-on-game-art-hq-d/','http://www.game-art-hq.com/52712/video-game-characters-on-game-art-hq-e/','http://www.game-art-hq.com/52716/video-game-characters-on-game-art-hq-f/','http://www.game-art-hq.com/52721/video-game-characters-on-game-art-hq-g/','http://www.game-art-hq.com/52731/video-game-characters-on-game-art-hq-h/','http://www.game-art-hq.com/52738/video-game-characters-on-game-art-hq-i/','http://www.game-art-hq.com/52742/video-game-characters-on-game-art-hq-j/','http://www.game-art-hq.com/52750/video-game-characters-on-game-art-hq-k/','http://www.game-art-hq.com/52759/video-game-characters-on-game-art-hq-l/','http://www.game-art-hq.com/52768/video-game-characters-on-game-art-hq-m/','http://www.game-art-hq.com/52774/video-game-characters-on-game-art-hq-n/','http://www.game-art-hq.com/52780/video-game-characters-on-game-art-hq-o/','http://www.game-art-hq.com/52784/video-game-characters-on-game-art-hq-p/','http://www.game-art-hq.com/52798/video-game-characters-on-game-art-hq-q/','http://www.game-art-hq.com/52802/video-game-characters-on-game-art-hq-r/','http://www.game-art-hq.com/52809/video-game-characters-on-game-art-hq-s/','http://www.game-art-hq.com/52820/video-game-characters-on-game-art-hq-t/','http://www.game-art-hq.com/52841/video-game-characters-on-game-art-hq-u/','http://www.game-art-hq.com/52845/video-game-characters-on-game-art-hq-v/','http://www.game-art-hq.com/52851/video-game-characters-on-game-art-hq-w/','http://www.game-art-hq.com/52856/video-game-characters-on-game-art-hq-x/','http://www.game-art-hq.com/52866/video-game-characters-on-game-art-hq-y/','http://www.game-art-hq.com/52578/video-game-characters-on-game-art-hq-z/']
    for base in pages:
        print("Base URL is ", base)
        driver.get(base)

        images = driver.find_elements_by_tag_name('img')
        for image in images:
            print(image.get_attribute('src'))
            urllib.request.urlretrieve(image.get_attribute('src'), "data/"+str(count)+".png")
            count += 1

        print("##############################SLEEPING################################")
        time.sleep(10)


    driver.quit()


if __name__== "__main__":
    main()
