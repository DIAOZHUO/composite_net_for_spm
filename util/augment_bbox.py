import albumentations as A
import cv2
import os
from util.file_io import get_datasets_filename
import matplotlib.pyplot as plt

transform = A.Compose([
    A.Flip(),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.Resize(height=256, width=256),
    A.RandomSizedCrop(height=256, width=256, min_max_height=(200, 256)),
    A.RandomGamma(gamma_limit=(80, 120), always_apply=False, p=0.8),
    # A.Downscale(scale_min=0.8, scale_max=0.999, p=0.8),
    A.Resize(height=256, width=256),
], bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"]))



def argument_data(data_path_name, argument_time=5):
    """
    :param data_path_name: need one N.png and one N.txt file, the value should pass as N
    :param argument_time:
    :return:
    """
    folder_path = os.path.dirname(data_path_name)
    os.makedirs(os.path.join(folder_path, "argument_data"), exist_ok=True)
    folder_path = os.path.join(folder_path, "argument_data")
    base_name = os.path.basename(data_path_name)
    print(base_name)
    image = cv2.imread(data_path_name+".png")
    f = open(data_path_name+'.txt', 'r')
    anno_lists = f.readlines()
    f.close()

    bboxes = []
    class_labels = []

    for anno_list in anno_lists:
        anno_list = anno_list.split()
        class_labels.append(anno_list[0])
        anno_list = [float(i) for i in anno_list[1:]]
        bboxes.append(anno_list)

    for i in range(argument_time):
        try:
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']
            # plt.imshow(transformed_image)
            # plt.show()

            if len(transformed_bboxes) > 0:
                cv2.imwrite(os.path.join(folder_path, "A%d_%s.png"%(i, base_name)), transformed_image)
                f = open(os.path.join(folder_path, "A%d_%s.txt"%(i, base_name)), 'w')


                for i, transformed_bbox in enumerate(transformed_bboxes):
                    f.write("{} {} {} {} {}\n".format(transformed_class_labels[i], transformed_bbox[0],
                        transformed_bbox[1],transformed_bbox[2],transformed_bbox[3]))
                f.close()
        except:
            print("error occur in file: %s on A%d" % (base_name, i))



if __name__ == '__main__':
    data_folder = "E:\PythonProjects\yolov8\si77_adsorption_detection/train\obj_train_data/"
    datasets = get_datasets_filename(data_folder, ext="txt")
    for it in datasets:
        argument_data(data_folder+it, argument_time=5)




