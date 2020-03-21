'''
 AUTHORS:
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>
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
#from io import ImageIO
from datetime import datetime
import pandas as pd
import requests
import time
import urllib
import urllib.request
from string import ascii_lowercase



def main():
    print("Hello World, From The Video Game Scraper!")
    options = Options()
    options.add_argument("no-sandbox")
    #options.add_argument("headless")
    options.add_argument('--no-proxy-server')
    options.add_argument("start-maximized")
    options.add_argument("window-size=1900,1080");
    driver = webdriver.Chrome(options=options, executable_path="chromedriver")
    driver.set_page_load_timeout(15)
    count = 0
    for c in ascii_lowercase:
        base='http://www.game-art-hq.com/52008/the-video-game-character-database-letter-'+c+'/'
        driver.get(base)
        time.sleep(10)

        actions = ActionChains(driver)
        find = driver.find_elements_by_tag_name('img')
        actions.key_down(Keys.CONTROL).click(find).key_up(Keys.CONTROL).perform()

        driver.switch_to.window(driver.window_handles[-1])
        driver.get("https://stackoverflow.com")
        print("New Page?")

        images = driver.find_elements_by_tag_name('img')
        for image in images:
            print(image.get_attribute('src'))
            urllib.request.urlretrieve(image.get_attribute('src'), "data/"+str(count)+".png")
            count += 1
        time.sleep(10)

    driver.quit()


if __name__== "__main__":
    main()
