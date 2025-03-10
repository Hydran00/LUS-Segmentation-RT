import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import tensorflow as tf
from time import perf_counter
from cv_bridge import CvBridge
import pandas as pd
from src import Models as models
from src.DataLoader import DataLoader
from src import Visualise as vis

# MODEL PARAMETERS
CLASS_NAMES = ['Ribs', 'Pleural line', 'A-line', 'B-line', 'B-line confluence']
CLASSES = {i+1: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}  # map indices to classes
CMAP = vis.plt.cm.tab10.colors

class UltrasoundSegmentationNode(Node):
    def __init__(self):
        super().__init__('ultrasound_segmentation')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/screen',  # Change this topic name as needed
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning
        self.bridge = CvBridge()

        # Load model
        self.model = models.unet((256, 256, 1), 6, filters=[32, 64, 128, 256, 512])
        self.model.load_weights('model_lus.h5')
        self.dl = DataLoader(d_transforms={
            'crop': (15,41,226,330),
            'resize': (256, 256),
            'one-hot': True,
        })
        self.overlay = True
        self.overlay_transparency = 0.5
        self.d_times = {'preprocessing': [], 'inference': [], 'display': [], 'total': []}

        # Define visualization windows
        cv2.namedWindow("Model output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Model output", 1920, 1080)
        cv2.setMouseCallback('Model output', self.click_event)
        cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)
        legend = vis.create_cv2_legend(CLASSES.values(), cmap=[tuple(int(i*255) for i in c) for c in CMAP])
        cv2.imshow("Legend", legend)

    def image_callback(self, msg):
        self.get_logger().info('Received image')
        t0 = perf_counter()
        
        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #crop image
        (x1,y1,x2,y2) = (15,41,226,330)
        frame = frame[y1:y2, x1:x2]

        frame0 = frame.copy()
        # cv2.waitKey(0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = tf.convert_to_tensor(frame)[tf.newaxis, :, :, tf.newaxis]
        frame = self.dl.resize_and_rescale(frame)

        t1 = perf_counter()
        
        # Run inference
        pred_mask = self.model.predict(frame, verbose=0)[0]
        pred_mask = pred_mask.argmax(axis=2)
        pred_mask = vis.seg_to_rgb(pred_mask, cm=CMAP)
        pred_mask = cv2.resize(pred_mask.astype('float32'), (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)

        # Overlay mask
        if self.overlay:
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)
            frame = cv2.addWeighted(frame0 / 255, 1.0, pred_mask, self.overlay_transparency, 0, dtype=cv2.CV_32F)

        t2 = perf_counter()

        # Display output
        cv2.imshow('Model output', frame)
        key = cv2.waitKey(1)
        # if key == ord('q'):
        #     rclpy.shutdown()
        # elif key == ord('w'):
        #     self.overlay_transparency = min(self.overlay_transparency + 0.1, 1.0)
        # elif key == ord('s'):
        #     self.overlay_transparency = max(self.overlay_transparency - 0.1, 0.0)

        self.d_times['preprocessing'].append(t1-t0)
        self.d_times['inference'].append(t2-t1)
        self.d_times['display'].append(perf_counter()-t2)
        self.d_times['total'].append(perf_counter()-t0)

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.overlay = not self.overlay

    def save_timing_performance(self):
        df_scores = pd.DataFrame(self.d_times)
        df_scores.to_csv('timing_performance.csv')
        print(df_scores[1:].describe())


def main(args=None):
    rclpy.init(args=args)
    node = UltrasoundSegmentationNode()
    rclpy.spin(node)
    node.save_timing_performance()
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
