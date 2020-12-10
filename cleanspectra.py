from model import *
from prepocessimg import *
from os import argv
from astropy import fits

size = (1024, 256)

file = argv[1]

img, header = extract(file)


model = build_model(size)
model.load_weights('model_spectra_noise_removing.hdf5')

img = img.reshape((1,)+size+(1,))
precited = model.predict(img).reshape(size)

fits.writeto('cleaned.fits', precited, header)