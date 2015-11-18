from sys import maxsize
from Queue import Queue as Q
from PIL import Image
import numpy

def find_first_white(pixels):
    for i in range(28):
        for j in range(28):
            if pixels[j,i] <= 100:
                return [j,i]
    return [-1,-1]

def pix_density(img):
    density = 0
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[j,i] > 100:
                density += 1
    return density

def density_ul(img):
    density = 0
    pixels = img.load()
    for i in range(img.size[0]/2):
        for j in range(img.size[1]/2):
            if pixels[j,i] > 100:
                density += 1
    return density

def density_ur(img):
    density = 0
    pixels = img.load()
    for i in range(img.size[0]/2):
        for j in range(img.size[1]/2, img.size[1]):
            if pixels[j,i] > 100:
                density += 1
    return density

def density_bl(img):
    density = 0
    pixels = img.load()
    for i in range(img.size[0]/2, img.size[0]):
        for j in range(img.size[1]/2):
            if pixels[j,i] > 100:
                density += 1
    return density
def density_br(img):
    density = 0
    pixels = img.load()
    for i in range(img.size[0]/2, img.size[0]):
        for j in range(img.size[1]/2, img.size[1]):
            if pixels[j,i] > 100:
                density += 1
    return density

def hgt_to_wdth(img):
    l_hgt = maxsize
    l_wdth = maxsize
    m_hgt = 0
    m_wdth = 0
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if pixels[j,i] > 100:
                if j < l_wdth:
                    l_wdth = j
                if j > m_wdth:
                    m_wdth = j
                if i < l_hgt:
                    l_hgt = j
                if i > m_hgt:
                    m_hgt = i
    height = float(m_hgt - l_hgt)
    width = float(m_wdth - l_wdth)
    return float(width/height)

def grey_out(pixels, x, y):
    pixels[x,y] = 150
    if (x+1 <=27) and pixels[x+1,y] <= 100:
        grey_out(pixels, x+1, y)
    if (y+1 <=27) and pixels[x,y+1] <= 100:
        grey_out(pixels, x, y+1)
    if (x-1 >= 0) and pixels[x-1, y] <= 100:
        grey_out(pixels, x-1, y)
    if (y-1 >= 0) and pixels[x,y-1] <= 100:
        grey_out(pixels, x, y-1)
    return pixels

def num_holes(img):
    pixels = img.load()
    num_holes = 0
    fw = find_first_white(pixels)
    while fw != [-1,-1]:
        pixels = grey_out(pixels, fw[0], fw[1])
        num_holes += 1
        fw = find_first_white(pixels)
    return num_holes

def horiz_symmetry(img):
    pixels = img.load()
    symmetry = 0
    for i in range(img.size[0]/2):
        for j in range(img.size[1]):
            if pixels[j,i] > 100 and pixels[28-j,i]:
                symmetry += 1
    return symmetry

def num_intersections(img):
    pixels = img.load()
    intersections = 0
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if j is not 0:
                if pixels[j,i] != pixels[j-1,i]:
                    intersections += 1
    return intersections






