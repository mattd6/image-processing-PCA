import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from PIL import Image
import cv2
import glob

## Linearize an image
def lin_image(base_image):
    (H,W,ch) = np.shape(base_image)    # Get the Shape
    lin_image = base_image.reshape(H*W).squeeze()  # linearize
    return lin_image

## Unlinearize an pre-linearize image
def unlin_image(in_image,H,W):
    L = np.shape(in_image)[0]   # Get the Shape
    # Verify the possible size of the image
    if L != H*W:
        raise ValueError('The length of the image does not correspond to the Height and width input')
    # unlinearize
    base_image = in_image.reshape(H,W).squeeze()
    return base_image

## Compute the PCA of the image
def PCAImage(image,n_components=1):
    # Get the size of the image
    (H, W) = np.shape(image)
    #print(h, rest)
    # Create PCA Object from scikit-learn library
    sklearn_pca = PCA(n_components=n_components)

    # Apply the PCA to the linearize image
    lin_pca = sklearn_pca.fit_transform(image)
    #lin_pca = lin_image(image)

    # transform from linearize image to normal image
    #out_pca = unlin_image(lin_pca, H, W)
    out_pca = sklearn_pca.inverse_transform(lin_pca)
    # casting the output image to uint8 format (0-255)
    out_pca = np.uint8((out_pca - np.min(out_pca)) * 255 / (np.max(out_pca) - np.min(out_pca)))
    return out_pca


## print the example of the PCA computation
def imageExample():

    df = glob.glob("<put image path here>")

    input_image = []

    img = cv2.imread(df[0], -1)
    roi = cv2.selectROI(img.astype(np.uint8))
    #print(roi)

    # print(rec)

    for fn in df:
        #reshaped_img = np.reshape(cv2.imread(fn), -1)
        img = cv2.imread(fn, -1)
        # crop = img[(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        x, y, w, h = roi
        crop = img[int(y):int(y+h), int(x): int(x+w)]
        # reshaped_img = np.reshape(crop, -1)
        #reshaped_img = np.reshape(img, -1)

        pca_image = PCAImage(crop, n_components=1)
        input_image.append(pca_image)

    #rgb_image = np.dstack(input_image)

    #rgb_array = []
    rgb_array = np.array([roi[3], roi[2], 3], dtype=np.uint8)

    for i in range(3):
        new_img = input_image[i].reshape(roi[3], roi[2], -1)
        #print(new_img.shape)
        rgb_array[i] = new_img

    #rgb_array = np.array([img.shape[0], img.shape[1],3], dtype=np.uint8)
    """rgb_array[:, :, 0] = r
    rgb_array[:, :, 1] = g
    rgb_array[:, :, 2] = b"""

    if input_image is None:
        print('Could not open or find the image: ')
        exit(0)

    # Show the tree image to compare
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.title('Input image')
    # plt.imshow(input_image)

    plt.subplot(2,2,4)
    plt.title('PCA image')
    plt.imshow(rgb_array)
    plt.show()

## main program
if __name__ == '__main__':
    imageExample()
