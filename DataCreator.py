'''
 AUTHORS: (in no particular order)
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>

 PURPOSE:
        Take the RAW data from CharacterScraper/data and resizes them to 100 x 100 pixels
        as well as
'''
import os
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd



# Create the Column names
columnNames = list()
for i in range(6400):
    pixel = 'pixel'
    pixel += str(i)
    columnNames.append(pixel)

#Create Dataframe
train_data = pd.DataFrame(columns = columnNames)
train_labels = pd.DataFrame(columns = ['image#'])
'''
# Get data from CharacterScraper/data and then Grayscale it and resize to 80x80
directory = os.fsencode("./CharacterScraper/data")
count = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        print(".png file found")
        print(file.decode("utf-8"))
        image = PIL.Image.open("./CharacterScraper/data/"+ str(file.decode("utf-8"))).convert('LA')
        width, height = image.size
        print(width, height)
        if width != 80:
            #Resize image to 100x100 pixels
            new_image = image.resize((80,80))
            image = new_image
        image.save('./Data/train_data/grayscale_images/'+str(count)+".png")
        count += 1
        continue
    else:
        continue
'''
#Create CSV file for Grayscaled images
directory = os.fsencode("./Data/train_data/grayscale_images")
i = 1
for img_name in os.listdir(directory):
    print("At "+ str(img_name.decode("utf-8")))
    train_labels.loc[i] = str(img_name.decode("utf-8"))
    img = Image.open("./Data/train_data/grayscale_images/" + str(img_name.decode("utf-8")))
    rawData = img.load()
    data = []
    for y in range(80):
        for x in range(80):
            data.append(rawData[x,y][0])
    k = 0
    train_data.loc[i] = [data[k] for k in range(6400)]
    i+=1

train_data.to_csv("./Data/train_data/train_converted.csv",index = False)
train_labels.to_csv("./Data/train_data/train_labels.csv",index = False)
