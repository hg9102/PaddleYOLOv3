import sys
import numpy as np
import time
import paddle.fluid as fluid
import os

from IPython.display import display
from PIL import Image, ImageDraw

from config import train_parameters, init_train_parameters

init_train_parameters()
ues_tiny = train_parameters['use_tiny']
yolo_config = train_parameters['yolo_tiny_cfg'] if ues_tiny else train_parameters['yolo_cfg']

target_size = yolo_config['input_size']
anchors = yolo_config['anchors']
anchor_mask = yolo_config['anchor_mask']
label_dict = [
    "111ti",
    "111to",
    "111w",
    "222ti",
    "222to",
    "222w",
    "333ti",
    "333to",
    "333w",
    "444ti",
    "444to",
    "444w",
    "555ti",
    "555to",
    "555w",
    "666ti",
    "666to",
    "666w",
    "777ti",
    "777to",
    "777w",
    "888ti",
    "888to",
    "888w",
    "999ti",
    "999to",
    "999w"
]

class_dim = train_parameters['class_dim']
print("label_dict:{} class dim:{}".format(label_dict, class_dim))
place = fluid.CPUPlace()
exe = fluid.Executor(place)
path = train_parameters['freeze_dir']
[inference_program, feed_target_names,
    fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)

print("targets {0}".format(fetch_targets))


def draw_bbox_image(img, boxes, scores, labels, save_name):
    """
    给图片画上外接矩形框
    :param img:
    :param boxes:
    :param save_name:
    :param labels
    :return:
    """

    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        print("{0} {1} {2}".format(
            label_dict[int(label)], str(score)[:6], box))
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        draw.rectangle((xmin, ymin, xmax, ymax), None, 'red')
        draw.text((xmin + 5, ymin), str(score)[:6], (0, 0, 0))
        draw.text((xmin + 5, (ymin + ymax) / 2),
                  label_dict[int(label)], (0, 150, 0))
    img.save(save_name)
    display(img)


def resize_img(img, target_size):
    """
    保持比例的缩放图片
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize(target_size[1:], Image.BILINEAR)
    return img


def read_image(img_path):
    """
    读取图片
    :param img_path:
    :return:
    """
    origin = Image.open(img_path)
    img = resize_img(origin, target_size)
    resized_img = img.copy()
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin, img, resized_img


def infer(image_path):
    """
    预测，将结果保存到一副新的图片中
    :param image_path:
    :return:
    """
    origin, tensor_img, resized_img = read_image(image_path)
    input_w, input_h = origin.size[0], origin.size[1]
    image_shape = np.array([input_h, input_w], dtype='int32')
    # print("image shape high:{0}, width:{1}".format(input_h, input_w))
    t1 = time.time()
    batch_outputs = exe.run(inference_program,
                            feed={feed_target_names[0]: tensor_img,
                                  feed_target_names[1]: image_shape[np.newaxis, :]},
                            fetch_list=fetch_targets,
                            return_numpy=False)
    period = time.time() - t1
    print("predict cost time:{0}".format("%2.2f sec" % period))
    bboxes = np.array(batch_outputs[0])
    # print(bboxes)

    if bboxes.shape[1] != 6:
        print("No object found in {}".format(image_path))
        return
    labels = bboxes[:, 0].astype('int32')
    scores = bboxes[:, 1].astype('float32')
    boxes = bboxes[:, 2:].astype('float32')

    last_dot_index = image_path.rfind('.')
    out_path = image_path[:last_dot_index]
    out_path += '-result.jpg'
    draw_bbox_image(origin, boxes, scores, labels, out_path)


if __name__ == '__main__':
    image_name = sys.argv[1]
    files = os.listdir(image_name)
    for file in files:
        if not (file.startswith(".") or file.endswith("result.jpg")):
            infer(image_name + file)
    # image_path = image_name
    # infer(image_path)
