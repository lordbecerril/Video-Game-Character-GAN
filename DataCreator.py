'''
 AUTHORS: (in no particular order)
        Eric Becerril-Blas    <Github: https://github.com/lordbecerril>
        Itzel Becerril        <Github: https://github.com/HadidBuilds>
        Erving Marure Sosa    <Github: https://eems20.github.io/>

 PURPOSE:
        Take the RAW data from CharacterScraper/data and resizes them to 100 x 100 pixels
'''
import os
import PIL
from PIL import Image

directory = os.fsencode("./CharacterScraper/data")
count = 0
not100 = 0
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        print(".png file found")
        print(file.decode("utf-8"))
        image = PIL.Image.open("./CharacterScraper/data/"+ str(file.decode("utf-8")))
        width, height = image.size
        print(width, height)
        if width != 100:
            #Resize the Sucka
            not100 += 1
            new_image = image.resize((100,100))
            image = new_image
        image.save('./Data/train_data/'+str(count)+".png")
        count += 1
        continue
    else:
        continue
print(not100)
