from resnet_v1_50_preprocessing import *

def eval_input(iter, eval_image_dir, eval_image_list, class_num, eval_batch_size):
  images = []
  labels = []
  line = open(eval_image_list).readlines()
  for index in range(0, eval_batch_size):
    curline = line[iter * eval_batch_size + index]
    [image_name, label_id] = curline.split(' ')
    image = cv2.imread(eval_image_dir + image_name)
    image = central_crop(image, 224, 224)
    image = mean_image_subtraction(image, MEANS)
    images.append(image)
    labels.append(int(label_id))
  lb = preprocessing.LabelBinarizer()
  lb.fit(range(0, class_num))
  labels = lb.transform(labels)
  return {"input": images, "labels": labels}


calib_image_dir = "../images/image224/"
calib_image_list = "../images/image500/calib.txt"
calib_batch_size = 1
def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()
    image = cv2.imread(calib_image_dir + calib_image_name)
    image = central_crop(image, 224, 224)
    image = mean_image_subtraction(image, MEANS)
    images.append(image)
  return {"input": images}

dump_image_dir = "../images/image224/"
dump_image_list = "../images/image224/tf_dump.txt"
dump_batch_size = 1
def dump_input(iter):
  images = []
  line = open(dump_image_list).readlines()
  for index in range(0, dump_batch_size):
    curline = line[iter * dump_batch_size + index]
    dump_image_name = curline.strip()
    image = cv2.imread(dump_image_dir + dump_image_name)
    image = central_crop(image, 224, 224)
    image = mean_image_subtraction(image, MEANS)
    images.append(image)
  return {"input": images}
