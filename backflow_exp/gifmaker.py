import imageio


images = []
for t in range(40):
    images.append(imageio.imread('circles/' + str(t+1) + '.0gersh.png'))
imageio.mimsave('circles/circles.gif', images, duration=1)
