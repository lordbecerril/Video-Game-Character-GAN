Docker container which runs the character scraper from http://www.game-art-hq.com <br/>

Soon will be Dockerized... <br/>


So far it is a python script with chromedriver in the same directory. <br/>

Things to change:<br/>
- Remove Hardcoded URLS (All URLS are the same. There is a way to get the inner HTML hrefs of the Letter on the side panel) <br/>
- This script gets ALLLLL images from the page it is on... this includes the side panel images so need to go through and clean dataset. There is a way to just get images located in tbody of the HTML
- Just saves images, not sure if we need to label them...
