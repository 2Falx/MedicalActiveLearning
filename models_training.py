import train

def main():
    train_images_path = "train/masked_train"
    train_input_path = "train/patched_images"
    train_CHASE_DB1_path = "CHASE_DB1/patched_images"
    test_path = "train/patched_images_test"
    active_learning = True
    model_name = 'pnet' # 'pnet', 'vgg', 'resnet'
    
    if active_learning:
        labels = train.train_active_learning(train_input_path, train_CHASE_DB1_path,
                                             test_path, num_iterations=0, metrics="least_confidence",
                                             use_second_dataset=False, method="canny",model=model_name)
    else:
        # get the labels for segmentation through kmeans/canny and eventually an additional dataset
        labels = train.train_whole_dataset(train_input_path, train_CHASE_DB1_path, test_path,
                                           use_second_dataset=False, method="canny", model=model_name,
                                           undersamling=False)
    
    # Train segmentation model using the labels from the previous training
    train.segnet(train_images_path, labels)

    print("End")

if __name__ == "__main__":
    main()