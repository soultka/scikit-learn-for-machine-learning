import mahotas as mh
from mahotas.features import surf

image = mh.imread('mandrill.png', as_grey=True)
print('The first SURF descriptor:\n', surf.surf(image)[0])
print ('Extracted %s SURF descriptors' % len(surf.surf(image)))
