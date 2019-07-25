import imageio
import numpy


images = []
for t in numpy.arange(14, 40, 1):
    images.append(imageio.imread('circles/' + str(t+1) + '.0gersh.png'))
imageio.mimsave('circles/circles.gif', images, duration=1)
