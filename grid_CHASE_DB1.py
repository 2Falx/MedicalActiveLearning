from utility import *
import imageio as io
from pathlib import Path

def main():
    masked_train_path = "CHASE_DB1/images/"
    label_path = "CHASE_DB1/labels/"
    patch_size = 32
    saved_path = "CHASE_DB1/patched_images/"
    input_train_images = [item for item in os.listdir(masked_train_path)]
    grid_path = "CHASE_DB1/images_with_grid/"

    # Create folders if they don't exist already
    Path('CHASE_DB1/patched_images/').mkdir(parents=True, exist_ok=True)
    Path('CHASE_DB1/images_with_grid/').mkdir(parents=True, exist_ok=True)

    label_train_images = [item for item in os.listdir(label_path)]

    for i, j in enumerate(input_train_images):
        image = io.imread(masked_train_path + j)
        label_image = io.imread(label_path + label_train_images[i])
        label_mat = np.array(label_image)
        # prob_mat = np.zeros(np.array(image).shape, dtype=np.float32)
        prob_mat = np.array(image)
        image_mat = np.array(image)
        x_dim, y_dim, z_dim = prob_mat.shape
        # getting the two dimension where there is still eye
        sum_mat = np.sum(image_mat, axis=2)
        x, y = np.where(sum_mat)
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        # calculate the number of patches in x and y direction
        num_of_x_patches = np.int(np.ceil((x_max - x_min)/patch_size))
        num_of_y_patches = np.int(np.ceil((y_max - y_min)/patch_size))
        for m in range(num_of_x_patches):
            for n in range(num_of_y_patches):
                patch_start_x = x_min + patch_size * m
                patch_end_x = x_min + patch_size * (m + 1)
                patch_start_y = y_min + patch_size * n
                patch_end_y = y_min + patch_size * (n + 1)
                # Modify the last patch in the row if it is out of bounds
                if patch_end_x >= x_dim:
                    patch_end_x = x_max
                    patch_start_x = x_max - patch_size
                if patch_end_y >= y_dim:
                    patch_end_y = y_max
                    patch_start_y = y_max - patch_size
                prob_mat[patch_start_x: patch_end_x, patch_start_y] = 1
                prob_mat[patch_start_x: patch_end_x, patch_end_y] = 1
                prob_mat[patch_start_x, patch_start_y: patch_end_y] = 1
                prob_mat[patch_end_x, patch_start_y: patch_end_y] = 1
                toSave = image_mat[patch_start_x: patch_end_x, patch_start_y:patch_end_y]

                # I generate the labels to automatize the work of the oracle
                gridToSave = label_mat[patch_start_x: patch_end_x, patch_start_y:patch_end_y]

                # If you want to have the different patches (with and without vessels in different paths)
                if 255 in gridToSave:
                    label="vessel"
                    path_to_save = saved_path + "class_1/"
                else:
                    label="no_vessel"
                    path_to_save = saved_path + "class_0/"

                # If you want to remove all the images which are totally black
                if np.max(toSave) != 0:
                    createAndSaveImage(toSave, saved_path + f"{label}_{m}_{n}_{j[:-3]}jpg")

        createAndSaveImage(prob_mat, grid_path + j)
        print("Patched generated")


if __name__ == "__main__":
    main()