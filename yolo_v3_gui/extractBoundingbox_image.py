import cv2

image = cv2.imread(r"Z:\Golden_Lab_Users\Choong_JJ\yolo_socks\Source_Images\Test_Image_Detection_Results\171_out.png")

img = image[686:1170,102:943]

cv2.imwrite((r"Z:\Golden_Lab_Users\Choong_JJ\yolo_socks\Source_Images\Test_Image_Detection_Results\171_crop.png"),img)