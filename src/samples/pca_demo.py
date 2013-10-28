'''
Demonstrates the following functions from the svo_util library:
1. normalize_data()
2. scatter3d
3. scaleto8bit
4. reconstructPCA

Requires scikits.learn (sklearn) for the olivetti faces data set,
and pyvision for displaying images and montages.
'''

from sklearn.datasets.olivetti_faces import *
import svo_util
import scipy as sp
import scipy.linalg as LA
import pyvision as pv

#fetch the data: Olivetti Faces, 64x64 pixels of gray-scale faces
#there are 10 face images per person, 40 people (400 images) total
x = fetch_olivetti_faces()

D = x['data']	#data matrix is 400 rows by 4096 pixels
L = x['target']  #these are integers indicating which person is in the image

#mean center and scale features Dn = (D-mean(D))/stdv(D)
Dn,means,stds = svo_util.normalize_data(D) 

#first, we'll show an image montage of the first 30 faces (3 people, 10 faces each)
img_list = [ pv.Image( x['images'][i].T) for i in range(400) ]
montage = pv.ImageMontage( img_list[0:30], layout=(3,10))
montage.show(delay=1, pos=(20,20) )

#PCA of the first 30 images, to yield transformed data T, and weights W
# computed by a thin svd. Since Dn is 400x4096, s is limited to 400 nonzero values
U,s,Vt = LA.svd(Dn[0:30,:])
W = Vt.T	#weights come from the right singular vectors when data samples are rows
T = Dn[0:30,:].dot(W)  #transform data using pca 'weights'

#Here we will show a 3D plot of the first 3 people (30 images) projected
# onto the first 3 PCA components. I.e., each face image is now a single 3D point
# in PCA space. This visualization shows that even with drastic dimensionality
# reduction, enough information is captured to discriminate between different faces
XYZ = T[:,0:3] #just the first 3 dimensions of the 30 points
svo_util.scatter3d(XYZ, labels=L[0:30], colors=['red','blue','green'])

#what do the principal components look like?
w_imgs = []
for i in range(20): #we will look at the first 20 "eigenfaces"
	w = W[:,i]
	ws = svo_util.scaleTo8bit(w)  #make values in range 0-255 so we can visualize as image
	wsi = ws.reshape(64,64).T  #change vector of 4096 values back to 64x64 pixel image
	w_imgs.append( pv.Image(wsi) )

montage2 = pv.ImageMontage( w_imgs, layout=(5,4) )
montage2.show("Top Principal Components", delay=1, pos=(20,200) )

#reconstruct first 30 images using only top N components (i.e., eigenfaces).
# reconstructPCA is a function that inverts the data transform: Xhat = T*W'
# but can limit the reconstruction to the first N components: Xhat = T[:,0:N]*W[:,0:N]'
# It also 'de-normalizes' the reconstruction by multiplying back the standard deviations
# and adding back the means. The result should look very similar to the input image
# when enough components are used. If all components are used, the images should be
# essentially identical within numerical precision errors.
reconstructed_data = []
for p in (3,10,20):
	Dhat = svo_util.reconstructPCA(T, W, components=p,  means=means, stds=stds)
	reconstructed_data.append( (p,Dhat) )

#show each reconstruction as montage
# notice that the reconstruction using only 3 components is blurrier,
# and also notice the eyes in the 3rd subject can't be reconstructed
# with only 3 components.
for idx,(p,Dhat) in enumerate( reconstructed_data):
	img_list = [ pv.Image( Dhat[i,:].reshape(64,64).T ) for i in range(30) ]
	imontage = pv.ImageMontage( img_list, layout=(3,10) )
	imontage.show("Reconstruction %d"%p, delay=1, pos=(300,200+idx*175))
	

#SYNTHETIC FACES
#construct new faces from random x,y,z coordinates in the PCA dimensions
# sampled uniformly within the ranges of each dimension
xr = [ min(XYZ[:,0]), max(XYZ[:,0]) ]
yr = [ min(XYZ[:,1]), max(XYZ[:,1]) ]
zr = [ min(XYZ[:,2]), max(XYZ[:,2]) ]

new_faces = []
for i in range(10):
	tmpx = sp.random.sample()*(xr[1]-xr[0]) + xr[0]
	tmpy = sp.random.sample()*(yr[1]-yr[0]) + yr[0]
	tmpz = sp.random.sample()*(zr[1]-zr[0]) + zr[0]
	new_faces.append( (tmpx,tmpy,tmpz) )
	
NXYZ = sp.array(new_faces)
A = sp.vstack((XYZ,NXYZ))  #combine 'new faces' with existing
Lx = list(L[0:30]) + ( [9]*10 )  #label '9' means new face

svo_util.scatter3d(A, labels=Lx, colors=['red','blue','green','cyan'])
synthesized_faces = []
S = svo_util.reconstructPCA(NXYZ, W, components=3, means=means, stds=stds)
img_list2 = [ pv.Image( S[i,:].reshape(64,64).T ) for i in range(10) ]
imontage2 = pv.ImageMontage( img_list2, layout=(2,5) )
imontage2.show("Synthesized Faces", delay=0)


