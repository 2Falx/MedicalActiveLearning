import random
import numpy as np
import keras.callbacks
from Nets.pnet import *
from Nets.resnet import *
from Nets.vgg import *
from Nets.wnetseg import *
from utility import *
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from clustering import *
from canny import *
from reconstructor import *
from scipy.stats import entropy

def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["accuracy"]
    val_accuracy = val_accuracy + history.history["val_accuracy"]
    return losses, val_losses, accuracy, val_accuracy


def plot_history(losses, val_losses, accuracies, val_accuracies):
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()


def get_Xy(path, external_dataset):
    unfiltered_filelist = getAllFiles(path)
    vessel_list = [item for item in unfiltered_filelist]
    X = []
    y = []
    for el, en in enumerate(vessel_list):
        if external_dataset and not "no_vessel" in en: continue
        X.append(load_to_numpy(en))
        if "no_vessel" in en:
            y.append(0)
        else:
            y.append(1)
    lb = LabelBinarizer()
    X = np.array(X, dtype=np.float64)
    y = lb.fit_transform(y)
    return X, y, vessel_list


# use it if you want to have a balanced (but smaller) training set and test set (test set to fix; it should remain unbalanced)
def random_under_sampling(X, y):
    indices = np.array(range(len(y)))
    # non vessels are at the beginning
    indices_non_vessels = np.array(range(len(y[y==0])))
    indices_vessels = np.random.choice(indices, size=len(y[y==0]))
    indices_under_sampling = indices_non_vessels.tolist() + indices_vessels.tolist()
    return X[indices_under_sampling], y[indices_under_sampling]

# use it to enlarge the dataset (in particular, the smallest class) by copying samples of that class
def random_over_sampling(X, y):
    y_vessels = y[y==1]
    y_non_vessels = y[y==0]
    difference = len(y_vessels) - len(y_non_vessels)
    X_non_vessels = X[np.where(y==0)[0]]
    X_to_add = X_non_vessels
    for i in range(int(np.floor(difference / len(X_non_vessels)) - 1)):
        X_to_add = np.concatenate((X_to_add, X_non_vessels))
    len_remaining_samples_to_add = difference - len(X_to_add)
    index_remaining_samples_to_add = np.random.choice(X_non_vessels.shape[0], len_remaining_samples_to_add)
    X_to_add = np.concatenate((X_to_add, X_non_vessels[index_remaining_samples_to_add]))
    y_to_add = np.array([0] * len(X_to_add))
    X_over_sampled = np.concatenate((X,X_to_add))
    y_over_sampled = np.concatenate((y,y_to_add.reshape(len(y_to_add),1)))
    return X_over_sampled, y_over_sampled

def normalize(X):
    train_mean = np.mean(X)  # mean for data centering
    train_std = np.std(X)  # std for data normalization
    X -= train_mean
    X /= train_std
    return X, train_mean, train_std


def shuffle_data(X,y, file_names):
    indices = np.array(range(len(y)))
    random.shuffle(indices)
    file_names = np.array(file_names)
    return X[indices], y[indices], file_names[indices]


def train_whole_dataset(patch_dir, path_CD, test_path, use_second_dataset, method):
    np.random.seed(42)
    # X, y origin dataset + list of names of files (useful for reconstructing images at the end)
    X, y, file_names_train = get_Xy(patch_dir, external_dataset=False)
    if use_second_dataset:  # true if we want to add the second dataset
        X_CD_no_vessel, y_CD_no_vessel, file_names_CD = get_Xy(path_CD, external_dataset=True)
        # set file_names_CD to 0 since we will want to reconstruct original data only (i.e. we use this as a flag to ignore)
        file_names_CD = np.zeros((len(file_names_CD),))
        X = np.concatenate((X, X_CD_no_vessel))
        y = np.concatenate((y, y_CD_no_vessel))
        file_names_train = np.concatenate((file_names_train, file_names_CD))
    # Uncomment if you want to try undersampling
    # X, y = random_under_sampling(X, y)
    X_train, y_train, file_names_train = shuffle_data(X,y, file_names_train)
    X_train, train_mean, train_std = normalize(X_train)
    X_test, y_test, file_names_test = get_Xy(test_path, external_dataset=False)
    X_test -= train_mean
    X_test /= train_std

    # NOTE: training the whole dataset with the classification network was done to assess performances and comparing
    #       them with active learning. Here we are simulating the case where we have all the labels thanks to the
    #       oracle (human expert). So, if you just want to exploit that annotations (at patch level) and passing them
    #       to the KMeans or to Canny methods, you can skip this CNN.

    patch_size = 32
    # Choose the model you want
    model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    # model = get_vgg(patch_size)
    #
    print('Training model...')
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[
          keras.callbacks.ModelCheckpoint(
                "FullModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )
  
    plot_history(
        history.history["loss"],
        history.history["val_loss"],
        history.history["accuracy"],
        history.history["val_accuracy"],
    )

    y_pred = model.predict(X_test)
    y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

    accuracy_score_test = accuracy_score(y_test, y_pred_rounded)
    precision_score_test = precision_score(y_test, y_pred_rounded)
    recall_score_test = recall_score(y_test, y_pred_rounded)
    f1_score_test = f1_score(y_test, y_pred_rounded)

    print(f"Accuracy on test: {accuracy_score_test}")
    print(f"Precision on test: {precision_score_test}")
    print(f"Recall on test: {recall_score_test}")
    print(f"f1 on test: {f1_score_test}")
    print(classification_report(y_test, y_pred_rounded))

    print()
    print('DONE')

    # Choose either kmeans or canny method to get a first approximation of pixel-level labels.
    if method == "kmeans":
        clustered_images = []
        for index, X_train_sample in enumerate(X_train):
            clustered_images.append(kmeans(X_train_sample, y_train.squeeze()[index]))
        images_to_rec = np.array(clustered_images)
    else:
        canny_images = []
        for index, X_train_sample in enumerate(X_train):
            canny_images.append(canny(X_train_sample, y_train.squeeze()[index]))
        images_to_rec = np.array(canny_images)

    # reconstruct segmented image by patches
    reconstructed_images = reconstruct(images_to_rec, file_names_train)
    return reconstructed_images

# used for active learning
def shuffle_split_and_normalize(X,y,file_names, X_test_final, train_size):
    indices = np.array(range(len(y)))
    random.shuffle(indices)
    indices_train = np.random.choice(len(y), size=int(train_size*len(y)),replace=False)
    indices_test = np.setxor1d(indices, indices_train)
    X_train = X[indices_train]
    y_train = y[indices_train]
    file_names_train = np.array(file_names)[indices_train]
    X_test = X[indices_test]
    y_test = y[indices_test]
    file_names_test = np.array(file_names)[indices_test]
    train_mean = np.mean(X)  # mean for data centering
    train_std = np.std(X)  # std for data normalization
    X_train -= train_mean
    X_train /= train_std
    X_test -= train_mean
    X_test /= train_std
    X_test_final -= train_mean
    X_test_final /= train_std
    return X_train, X_test, y_train, y_test, file_names_train, file_names_test, X_test_final


def train_active_learning(patch_dir, path_CD, test_path, num_iterations, metrics, use_second_dataset, method):
    np.random.seed(42)
    X, y, file_names = get_Xy(patch_dir, external_dataset=False)
    if use_second_dataset:  # true if we want to add the second dataset
        X_CD_no_vessel, y_CD_no_vessel, file_names_CD = get_Xy(path_CD, external_dataset=True)
        # set file_names_CD to 0 since we will want to reconstruct original data only (i.e. we use this as a flag to ignore)
        file_names_CD = np.zeros((len(file_names_CD),))
        X = np.concatenate((X, X_CD_no_vessel))
        y = np.concatenate((y, y_CD_no_vessel))
        file_names = np.concatenate((file_names, file_names_CD))
    # X, y = random_under_sampling(X, y)
    # We start with a 15% of the samples
    X_test_final, y_test_final, _ = get_Xy(test_path, external_dataset=False)
    X_train, X_test, y_train, y_test, file_names_train, file_names_test, X_test_final = shuffle_split_and_normalize(X, y, file_names, X_test_final, train_size=0.15)
    

    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies = [], [], [], []

    patch_size = 32
    model = get_pnetcls(patch_size)
    # model = get_resnet(patch_size)
    # model = get_vgg(patch_size)

    print("Starting to train... ")
    history = model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[keras.callbacks.ModelCheckpoint(
                "ALModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )

    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    #  Active Learning iterations
    for iteration in range(num_iterations):
        y_pred = model.predict(X_test_final)
        y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

        accuracy_score_test = accuracy_score(y_test_final, y_pred_rounded)
        precision_score_test = precision_score(y_test_final, y_pred_rounded)
        recall_score_test = recall_score(y_test_final, y_pred_rounded)
        f1_score_test = f1_score(y_test_final, y_pred_rounded)

        print(f"Accuracy on test: {accuracy_score_test}")
        print(f"Precision on test: {precision_score_test}")
        print(f"Recall on test: {recall_score_test}")
        print(f"f1 on test: {f1_score_test}")
        print(classification_report(y_test_final, y_pred_rounded))

        # Uncertain values count fixed
        count_uncertain_values = 50

        if metrics == "least_confidence":
            most_uncertain_indeces = np.argsort(np.abs(model.predict(X_test) - 0.5), axis=0)
            most_uncertain_indeces = most_uncertain_indeces[:count_uncertain_values].flatten()

        elif metrics == "entropy":
            entropy_y = np.transpose(entropy(np.transpose(model.predict(X_test))))
            most_uncertain_indeces = np.argpartition(-entropy_y, count_uncertain_values - 1, axis=0)[
                                     :count_uncertain_values]

        print(f"X_train.shape: {X_train.shape}")
        print(f"X_test[most_uncertain_indeces, :, :, :].shape: {X_test[most_uncertain_indeces, :, :, :].shape}")

        # Get most uncertain values from test and add them into the train
        X_train = np.vstack((X_train, X_test[most_uncertain_indeces, :, :, :]))
        y_train = np.vstack((y_train, y_test[most_uncertain_indeces, :]))
        file_names_train = np.concatenate((file_names_train, file_names_test[most_uncertain_indeces]))

        # remove most uncertain values from test
        X_test = np.delete(X_test, most_uncertain_indeces, axis=0)
        y_test = np.delete(y_test, most_uncertain_indeces, axis=0)
        file_names_test = np.delete(file_names_test, most_uncertain_indeces, axis=0)

        # Then I compile again and train again the model
        model.compile(optimizer="SGD",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            callbacks=[keras.callbacks.ModelCheckpoint(
                    "ALModelCheckpoint.h5", verbose=1, save_best_only=True
                ),
            ],
        )

        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

    # End of AL iterations

    # Loading the best model from the training loop
    model = keras.models.load_model("ALModelCheckpoint.h5")

    y_pred = model.predict(X_test_final)
    y_pred_rounded = np.where(np.greater(y_pred, 0.5), 1, 0)

    accuracy_score_test = accuracy_score(y_test_final, y_pred_rounded)
    precision_score_test = precision_score(y_test_final, y_pred_rounded)
    recall_score_test = recall_score(y_test_final, y_pred_rounded)
    f1_score_test = f1_score(y_test_final, y_pred_rounded)

    print(f"Accuracy on test: {accuracy_score_test}")
    print(f"Precision on test: {precision_score_test}")
    print(f"Recall on test: {recall_score_test}")
    print(f"f1 on test: {f1_score_test}")
    print(classification_report(y_test_final, y_pred_rounded))

    # Now put together train and test and use pass them to kmeans/canny for segmentation
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train.squeeze(), model.predict(X_test).squeeze()))  # we generate predictions on the whole dataset -> we will use them in kmeans/canny
    file_names = np.concatenate((file_names_train, file_names_test))

    # Choose either kmeans or canny method to get a first approximation of pixel-level labels.
    if method == "kmeans":
        clustered_images = []
        for index, X_train_sample in enumerate(X):
            clustered_images.append(kmeans(X_train_sample, y[index]))
        images_to_rec = np.array(clustered_images)
    else:
        canny_images = []
        for index, X_train_sample in enumerate(X):
            canny_images.append(canny(X_train_sample, y[index]))
        images_to_rec = np.array(canny_images)

    # reconstruct segmented image by patches
    reconstructed_images = reconstruct(images_to_rec, file_names)
    return reconstructed_images

    # return the labels
    return model


def segnet(train_input_path, labels):
    unfiltered_filelist = getAllFiles(train_input_path)
    vessel_list = [item for item in unfiltered_filelist]
    images = []
    for en in vessel_list:
        images.append(cv2.cvtColor(cv2.imread(en), cv2.COLOR_RGB2GRAY))
    X = np.array(images)

    X_train = X[:16]
    X_test = X[16:]
    labels = labels[:16]
    num_channels = 1
    activation = 'relu'
    final_activation = 'sigmoid'
    optimizer = Adam
    lr = 1e-4
    dropout = 0.1
    loss = 'categorical_crossentropy'
    metrics = 'accuracy'
    model = get_wnetseg(608, num_channels, activation, final_activation,
                        optimizer, lr, dropout, loss, metrics)

    history = model.fit(
        X_train,
        labels,
        epochs=40,
        batch_size=2,
        validation_split=0.2,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "FullModelCheckpoint.h5", verbose=1, save_best_only=True
            ),
        ],
    )

    y_pred = model.predict(X_test)

    for i in range(4):
        path = f"results/image{i}"
        y_pred_c = y_pred[i].squeeze()
        y_pred_copy = y_pred_c.copy()
        y_pred_c[y_pred_copy >= 0.2] = 1
        y_pred_c[y_pred_copy < 0.2] = 0
        plt.imshow(y_pred_c, "gray")
        plt.savefig(path)
        plt.show()
