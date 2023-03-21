import numpy as np
import matplotlib.pyplot as plt

def reconstruct(clustered_images, file_names):

    # create a numpy array which will contain ordered patches
    # since we discarded some black patches we initialize this array with full zeros (black)
    # originally we had 19*19 = 361 patches (32x32 size) per image, and we have 16 images in train so:
    shape = (5776, 32, 32)  # gray images
    ordered_patches = np.full(shape, 0) # np.zeros(shape)
    # we iterate over the patches name and we put the correspondant clustered image in the array of ordered patches
    for i,file_name in enumerate(file_names):
        # if it is a data of the additional dataset we skip it -> we want to reconstruct original data only
        if file_name == '0.0':
            continue
        # pick the image id and the position
        data = []  # x,y,id
        file_name_extract = file_name.replace("_", " ")
        for word in file_name_extract.split():
            if word.isdigit(): data.append(int(word))
        ordered_patches[361 * (data[2]-21) + data[0]*19 + data[1]] = clustered_images[i]  # cv2.cvtColor(clustered_images[i], cv2.COLOR_RGB2GRAY)

    final_images = np.zeros((16,608,608))  # 16 images 608x608
    for iteration in range(16):
        for i in range(19): # along the rows
            for j in range(19):  # along the columns
                if j == 0:
                    toAttachH = ordered_patches[iteration*361 + i*19 + j]
                else:
                    toAttachH = np.hstack((toAttachH, ordered_patches[iteration*361 + i*19 + j]))
            if i == 0:
                toAttachV = toAttachH
            else:
                toAttachV = np.vstack((toAttachV, toAttachH))
        final_images[iteration] = toAttachV


    for image in final_images:
        plt.imshow(image,"gray")
        plt.show()
    return final_images