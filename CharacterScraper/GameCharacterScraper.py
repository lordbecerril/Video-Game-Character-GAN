# Author: Eric Becerril
# Don't Judge My code, I suck :'(
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from io import StringIO
from io import ImageIO
from datetime import datetime
import pandas as pd
import requests

# Base Website, where we want to go when the website opens up
base='http://www.game-art-hq.com/52008/the-video-game-character-database-letter-a/'

def navtopage(driver):
    '''
        This function navigates to the correct page
    '''
    i=0
    while i<5:
        '''
            This while loop ensures we make it to the page
        '''
        try:
            print("sending req #"+str(i))
            driver.get(base)
            WebDriverWait(driver,15).until(EC.element_to_be_clickable((By.XPATH,'//*[@id="post-52008"]/div[1]/div[3]/table[1]/tbody/tr[1]/td[1]/p[2]/span/a')))#try to click 'flow'
            break
        except TimeoutException:
            i+=1
    #Get the character table
    character_table = driver.find_element_by_xpath('//*[@id="post-52712"]/div[1]/div[3]/table[1]')
    #driver.find_element_by_xpath('//*[@id="post-52008"]/div[1]/div[3]/table[1]/tbody/tr[1]/td[1]/p[2]/span/a').send_keys('\n')#if I have made it to the above page, click the data link

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
    navtopage(driver)#navigate to the correct page
    driver.quit()

if __name__== "__main__":
    main()
