import logging as log

# change TF log_level
import os;
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import sys
from pathlib import Path
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image


class Demo:

    def __init__(self):
        log.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=log.DEBUG, stream=sys.stdout)
        log.info(f"Initializing...")
        self.input_size = 416
        self.bbox_tensors = []
        self.input_layer = tf.keras.layers.Input([self.input_size, self.input_size, 3])
        self.feature_maps = YOLOv3(self.input_layer)
        self.model = None
        # model.summary()
        return

    def init_model(self):
        log.debug("init model")
        for i, fm in enumerate(self.feature_maps):
            bbox_tensor = decode(fm, i)
            self.bbox_tensors.append(bbox_tensor)

        self.model = tf.keras.Model(self.input_layer, self.bbox_tensors)
        log.debug("load weights")
        utils.load_weights(self.model, "./model/yolov3.weights")
        return

    def run(self):
        image_path = "./images"
        images = ["kite.jpg", "chairs.jpg",
                  "street01.png", "street02.png",
                  "room-sofa-chairs.jpg"]
        # images = ["chairs.jpg"]
        conf = 0.25
        sigma = 0.45
        self.init_model()
        for img in images:
            img = Path(image_path, img).absolute()
            if not img.exists():
                log.error(f"image {img} not found")
                continue
            log.debug(f"Process image {img}")
            original_image = cv2.imread(str(img))
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image_size = original_image.shape[:2]
            image_data = utils.preprocess_image(np.copy(original_image), [self.input_size, self.input_size])
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            prev_time = time.time()
            # log.debug("Predict")
            # pred_bbox = self.model.predict(image_data)
            pred_bbox = self.model(image_data)
            exec_time = time.time() - prev_time
            # log.debug("Done")
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, self.input_size, conf)
            # log.debug(f"{bboxes.shape[0]} boxes before nms")
            bboxes = utils.nms(bboxes, sigma)
            # log.debug(f"{len(bboxes)} boxes after nms")

            image = utils.draw_bbox(original_image, bboxes)
            # image = Image.fromarray(image)
            # image.show()
            # image.close()
            result = np.asarray(image)
            info = "time: %.2f ms" % (1000 * exec_time)
            cv2.putText(result, text=info, org=(10, 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=1)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("result", result)
            if cv2.waitKey(10000) & 0xFF == ord('q'):
                break

        return


if __name__ == "__main__":
    a = Demo()
    a.run()