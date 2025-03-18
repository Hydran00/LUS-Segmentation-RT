import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
import copy
import datetime
# MODEL PARAMETERS
CLASS_NAMES = ['Ribs', 'Pleural line', 'A-line', 'B-line', 'B-line confluence']
# CLASS_NAMES = ['Ribs', 'Pleural line', 'A-line', 'B-line', 'B-line confluence']
CLASS_NAMES = ['Pleural line']
CLASSES = {i+1: CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}  # map indices to classes
CMAP = vis.plt.cm.tab10.colors

class UltrasoundSegmentationNode(Node):
    def __init__(self):
        super().__init__('ultrasound_segmentation')
        # makedir
        folder_name = "detections_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs("outputs/" + folder_name)
        self.base_path = "outputs/" + folder_name
        print("Saving detections of pleura to: ", self.base_path)
        self.result_idx = 0
        self.bridge = CvBridge()

        # Load model
        self.model = models.unet((256, 256, 1), 6, filters=[32, 64, 128, 256, 512])
        self.model.load_weights('model_lus.h5')
        self.dl = DataLoader(d_transforms={
            'crop': None, #(15,41,226,330),
            'resize': (256, 256),
            'one-hot': True,
        })
        self.overlay = True
        self.overlay_transparency = 0.7
        self.d_times = {'preprocessing': [], 'inference': [], 'display': [], 'total': []}

        # Define visualization windows
        cv2.namedWindow("Model output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Model output", 1920, 1080)
        cv2.setMouseCallback('Model output', self.click_event)
        cv2.namedWindow("Legend", cv2.WINDOW_NORMAL)
        legend = vis.create_cv2_legend(CLASSES.values(), cmap=[tuple(int(i*255) for i in c) for c in CMAP])
        cv2.imshow("Legend", legend)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/screen',  # Change this topic name as needed
            self.image_callback,
            1
        )

    def image_callback2(self, msg):
        self.get_logger().info('Received image')

        # Convert compressed image to OpenCV format
        # np_arr = np.frombuffer(msg.data, np.uint8)
        # image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # # Crop image (adjust as needed)
        # (x1, y1, x2, y2) = (27, 41, 200, 330)
        # image = image[y1:y2, x1:x2]

        # # Convert to Grayscale & Denoise
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # # Enhance Contrast
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # contrast_enhanced = clahe.apply(blurred)

        # # Thresholding (Highlight Brightest Regions)
        # _, thresh = cv2.threshold(contrast_enhanced, 100, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Thresholded", thresh)

        # # Morphological Closing with Small Kernel (To avoid over-connection)
        # kernel = np.ones((3,3), np.uint8)  # Smaller kernel to keep horizontal details
        # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # # Detect Edges
        # edges = cv2.Canny(closed, 50, 150)

        # # Find Contours
        # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
        # cv2.imshow("Contours", image)   

        # Define a minimum contour area threshold
        # MIN_CONTOUR_AREA = 250  # Adjust as needed

        # # Filter out small contours
        # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        # # Select Longest Contour (Assuming it's the Pleural Line)
        # if filtered_contours:
        #     longest_contour = max(filtered_contours, key=cv2.contourArea)

        #     # Approximate the contour with a rotated bounding rectangle
        #     rect = cv2.minAreaRect(longest_contour)
        #     box = cv2.boxPoints(rect)
        #     box = np.intp(box)

        #     # Approximate the contour with an ellipse (if enough points)
        #     if len(longest_contour) >= 5:
        #         ellipse = cv2.fitEllipse(longest_contour)

        #     # Draw bounding shapes
        #     bounding_image = image.copy()
        #     cv2.drawContours(bounding_image, [box], 0, (0, 255, 0), 2)  # Green rectangle
        #     if len(longest_contour) >= 5:
        #         cv2.ellipse(bounding_image, ellipse, (0, 0, 255), 2)  # Red ellipse

        #     cv2.imshow("Bounding Shapes", bounding_image)

        #     # Fit a 2nd-degree Polynomial (Quadratic Curve) to the Pleural Line
        #     x = longest_contour[:, 0, 0]
        #     y = longest_contour[:, 0, 1]
        #     poly_coeffs = np.polyfit(x, y, deg=2)
        #     poly_fit = np.poly1d(poly_coeffs)

        #     # Draw the Fitted Curve
        #     fitted_image = image.copy()
        #     for i in range(min(x), max(x)):
        #         cv2.circle(fitted_image, (i, int(poly_fit(i))), 1, (255, 0, 0), -1)

        #     cv2.imshow("Fitted Pleural Line", fitted_image)

        cv2.waitKey(1)
            # cv2.destroyAllWindows()
            # cv2.destroyAllWindows()
   
    def get_biggest_component(self, mask, img):
        # Ensure mask and img are the same size
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Find all connected components
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        detection = copy.deepcopy(img)
        result = False
        
        if stats.shape[0] == 1:
            return mask, detection, result

        # Find the largest component (excluding background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip the background label
        num_pixel = stats[largest_label, cv2.CC_STAT_AREA]


        if num_pixel > 3000 and num_pixel < 10000:
            print("Go to the next area: ", num_pixel)

            # Bounding box of the largest component
            x = stats[largest_label, cv2.CC_STAT_LEFT]
            y = stats[largest_label, cv2.CC_STAT_TOP]
            w = stats[largest_label, cv2.CC_STAT_WIDTH]
            h = stats[largest_label, cv2.CC_STAT_HEIGHT]

            # Convert coordinates to match the original image
            center_x = x + w // 2
            center_y = y + h // 2
            axes_x = w // 2
            axes_y = h // 2

            # Draw the red ellipse
            cv2.ellipse(detection, (center_x, center_y), (axes_x, axes_y), angle=0, 
                        startAngle=0, endAngle=360, color=(0, 0, 255), thickness=2)
            result = True

        # Create a new mask with only the largest component
        largest_mask = (labels == largest_label).astype(np.uint8)

        return largest_mask, detection, result


    def image_callback(self, msg):
        self.get_logger().info('Received image')
        t0 = perf_counter()
        
        # Convert compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # mask the green dot
        frame[77:92,111:130][:] = 0

        # cv2.imshow('Original', frame)
        # cv2.waitKey(0)
        
        #crop image
        # (x1,y1,x2,y2) = (26,77,450,734)
        
        (x1,y1,x2,y2) = (26,77,450,500)
        # (x1,y1,x2,y2) = (26,127,450,500)
        
        
        # (x1,y1,x2,y2) = (27, 41, 200, 330)
        frame = frame[y1:y2, x1:x2]

        # fit with zeros to reach a square shape
        if frame.shape[0] > frame.shape[1]:
            pad = (frame.shape[0] - frame.shape[1]) // 2
            frame = cv2.copyMakeBorder(frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
        elif frame.shape[1] > frame.shape[0]:
            pad = (frame.shape[1] - frame.shape[0]) // 2
            frame = cv2.copyMakeBorder(frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        
        # cv2.imshow('Next', frame)
        # cv2.waitKey(1)

        frame0 = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = tf.convert_to_tensor(frame)[tf.newaxis, :, :, tf.newaxis]
        frame = self.dl.resize_and_rescale(frame)

        # frame_cv = copy.deepcopy(frame)
        # frame_cv = frame_cv.numpy().squeeze()
        # frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_GRAY2BGR)
        # cv2.imshow('Preprocessed', frame_cv)
        # cv2.waitKey(1)

        t1 = perf_counter()
        
        # Run inference
        pred_mask_prob = self.model.predict(frame, verbose=0)[0]
        
        # convert from 1-hot encoded to 2d array
        pred_mask = pred_mask_prob[:,:,3] #  pred_mask_prob.argmax(axis=2)

        # draw the mask
        pred_mask[0:50,:] = 0
        # cv2.imshow('Mask', pred_mask * 255)
        # cv2.waitKey(0)
        pred_mask[250:,:] = 0


        # cover the starting area
        pred_mask = pred_mask > 0.005

        bool_mask = pred_mask.astype(np.uint8)
        # get the biggest mask from the prediction
        frame_vis = frame0 / 255
        pred_mask, detection, result = self.get_biggest_component(bool_mask, frame_vis)
        pred_mask_cv = copy.deepcopy(pred_mask) * 255
        # cv2.imshow('Biggest Mask', pred_mask * 255)
        # cv2.waitKey(1)

        pred_mask = vis.seg_to_rgb(pred_mask, cm=CMAP)
        pred_mask = cv2.resize(pred_mask.astype('float32'), (frame0.shape[1], frame0.shape[0]), interpolation=cv2.INTER_AREA)

        # Overlay mask
        if self.overlay:
            pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)
            frame_vis = cv2.addWeighted(frame0 / 255, 1.0, pred_mask, self.overlay_transparency, 0, dtype=cv2.CV_32F)

        t2 = perf_counter()

        # Display output
        cv2.imshow('Model output', frame_vis)
        cv2.waitKey(1)

        if result:
            cv2.imshow('Detection', detection)
            # dump results
            print("Dumping results")
            # detection = cv2.cvtColor(detection, cv2.COLOR_BGR2RGB)
            # pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2RGB)
            # frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            # frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(self.base_path + '/mask.png', pred_mask)
            # cv2.imwrite(self.base_path + '/frame.png', frame0)

            # move to cpu
            frame_vis = frame_vis * 255
            detection = detection * 255
            cv2.imwrite(self.base_path + '/frame_vis_' + str(self.result_idx) + '.png', frame_vis)
            cv2.imwrite(self.base_path + '/detection.png', detection)
            cv2.imwrite(self.base_path + '/frame.png', frame0)
            cv2.imwrite(self.base_path + '/mask.png', pred_mask_cv)
            self.result_idx += 1
            cv2.waitKey(1)

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
