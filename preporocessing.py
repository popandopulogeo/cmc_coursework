import numpy as np
from astropy,io import fits
from os import mkdir
from os.path import exists, join

def clip(data, rate):
	rate = (1-rate)/2
    borders = np.empty(2)
    np.quantile(data, [rate, 1-rate], out=borders)
    np.clip(data, borders[0], borders[1], data)
    return data

def normalize(data):
    data = (data - data.min())/(data.max()-data.min())
    return 1 - data

def extract(file):
	img, header = fits.getdata(file, header=True)
	img = normalize(clip(img, 0.95))

	return img, header


