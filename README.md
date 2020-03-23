# Video-Game-Character-GAN
UNLV CS 489 final project. We are creating a GAN to generate video game
characters

**Team Memebers:** <br/> 
- Itzel Becerril 
- Eric Becerril-Blas
- Erving Marure Sosa

## VGC_GAN.py
The main script that does all the work.

## DataCreator.py
Takes the RAW data from CharacterScraper/data and resizes them to 80 x 80 pixels. Saves them into the Data/rgb_images directory

## Data
This directory holds the processed and cleaned pixel data
### rgb_images
This holds the RGB images

### grayscale_images
This holds the grayscaled images 

### train_data
Holds the training data. 1432 images. each image is 80x80 pixels

## Character Scraper Directory
This directory holds the RAW data and the web scraper made to collect the RAW data
