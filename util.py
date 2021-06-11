# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 01:51:10 2021

@author: ginag
"""
import imageio
import re
import os

# for sorting the sequence of file like 1, 2, 3.... 10, 11
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def exportGif(filePath, imgName):
    images = []
    for filename in sorted_alphanumeric(os.listdir(filePath)):
        if imgName in filename:
            images.append(imageio.imread( filePath +"/" +filename))
    
    # create a folder
    # if not os.path.exists(filePath +"/gif"):
    #     os.makedirs(filePath +"/gif") 
    imageio.mimsave(filePath +"/" + imgName + ".gif", images, duration= 0.5)  #"/gif/"