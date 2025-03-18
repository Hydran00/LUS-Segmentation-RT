import cv2

base = "outputs/detections_20250318-170423/"
detection = cv2.imread(base + "detection.png")
cv2.imshow("detection", detection)

frame_vis = cv2.imread(base + "frame_vis.png")
cv2.imshow("frame_vis", frame_vis)

frame = cv2.imread(base + "frame.png")
cv2.imshow("frame", frame)

mask = cv2.imread(base + "mask.png")
cv2.imshow("mask", mask)
cv2.waitKey(0)

