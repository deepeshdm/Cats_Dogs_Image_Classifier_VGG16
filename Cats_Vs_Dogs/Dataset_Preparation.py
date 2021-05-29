
# Author @ Deepesh Mhatre

# NOTE : Change the dataset directory paths below if you are running the code locally on your pc.

# Takes input_shape parameter so that we can resize the training
# images to dimensions suitable to any CNN architecture.

def get_data(cnn_input_shape=(227,227)):
    from os import listdir
    import numpy as np
    import cv2

    # This list will contain all images in their matrix format.4000 images of cats & dogs each.
    # Shape : (8000, 200, 200, 3)
    images = []

    # This list will act as "y". [dog,cat] - [0,1] for cat / [1,0] for dog.
    # Shape : (8000, 2)
    labels = []

    # ----------------------------------------------------------

    # Getting names of all cat images in the directory.
    cat_image_names = listdir('C:/Desktop/Datasets/Image Dataset/dog vs cat/dataset/training_set/cats')

    for cat in cat_image_names:
        img = cv2.imread(
            'C:/Desktop/Datasets/Image Dataset/dog vs cat/dataset/training_set/cats/' + cat)

        # Resizing the image.
        resized_img = cv2.resize(img, cnn_input_shape)

        # Converting image to its matrix form.
        img_array = np.asarray(resized_img)

        images.append(img_array)
        labels.append([0, 1])

    # -------------------------------------------------------------

    # Getting names of all dog images in the directory.
    dog_image_names = listdir('C:/Users/dipesh/Desktop/Datasets/Image Dataset/dog vs cat/dataset/training_set/dogs')

    for dog in dog_image_names:
        img = cv2.imread(
            'C:/Users/dipesh/Desktop/Datasets/Image Dataset/dog vs cat/dataset/training_set/dogs/' + dog)

        # Resizing the image.
        resized_img = cv2.resize(img, cnn_input_shape)

        # Converting image to its matrix form.
        img_array = np.asarray(resized_img)

        images.append(img_array)
        labels.append([1, 0])

    # -----------------------------------------------------------------

    images = np.array(images)
    labels = np.array(labels)

    # SAVING IMAGES & LABELS AS NUMPY ARRAY.
    np.save("Cat_Dog_Images_224x224.npy",images)
    np.save("Cat_Dog_Labels.npy",labels)

    return images, labels
