# CharacterScraper
An html scraper script that will scrape the images from the following website: <br/>
http://www.game-art-hq.com/52702/the-game-art-hq/ <br/>
It will iterate through the pages and save the raw images in a directory


# CharacterScraper Docker file (COMING SOON)
Docker container which runs the character scraper from http://www.game-art-hq.com <br/>

Build container with `docker build -t character_scraper .`  <br/>

Run with `docker run character_scraper` <br/>

Things to change:<br/>
- Remove Hardcoded URLS (All URLS are the same. There is a way to get the inner HTML hrefs of the Letter on the side panel) <br/>
- This script gets ALLLLL images from the page it is on... this includes the side panel images so need to go through and clean dataset. There is a way to just get images located in tbody of the HTML. ~885 images had to be removed during the cleaning process.
- Just saves images, not sure if we need to label them...
