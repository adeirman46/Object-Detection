import cv2
import numpy as np

# Below function will read video imgs
cap = cv2.VideoCapture(0)

while True:
    read_ok, img = cap.read()
    img_bcp = img.copy()
 
    img = cv2.resize(img, (640, 480))
    # Make a copy to draw contour outline
    input_image_cpy = img.copy()
 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    # define range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
     
    # define range of green color in HSV
    lower_green = np.array([40, 20, 50])
    upper_green = np.array([90, 255, 255])
     
    # define range of blue color in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
 
    # define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
 
    # define range of orange color in HSV
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([20, 255, 255])
 
    # define range of cyan color in HSV
    lower_cyan = np.array([75, 50, 50])
    upper_cyan = np.array([105, 255, 255])
     
    # define range of magenta color in HSV
    lower_magenta = np.array([160, 50, 50])
    upper_magenta = np.array([180, 255, 255])
     
    # define range of grey color in HSV
    lower_grey = np.array([0, 0, 0])
    upper_grey = np.array([255, 50, 50])
     
    # define range of white color in HSV
    lower_white = np.array([200, 50, 50])
    upper_white = np.array([255, 255, 255])
 
    # create a mask for red color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # create a mask for green color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # create a mask for blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # create a mask for yellow color
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # create a mask for orange color
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    # create a mask for cyan color
    mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
    # create a mask for magenta color
    mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
    # create a mask for grey color
    mask_grey = cv2.inRange(hsv, lower_grey, upper_grey)
    # create a mask for white color
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
 
    # find contours in the red mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the green mask
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the blue mask
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the yellow mask
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the orange mask
    contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the cyan mask
    contours_cyan, _ = cv2.findContours(mask_cyan, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the magenta mask
    contours_magenta, _ = cv2.findContours(mask_magenta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the grey mask
    contours_grey, _ = cv2.findContours(mask_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # find contours in the white mask
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
 
    # draw contours on the input image
    for cnt in contours_red:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(input_image_cpy, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    for cnt in contours_green:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(input_image_cpy, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for cnt in contours_blue:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(input_image_cpy, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    for cnt in contours_yellow:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(input_image_cpy, 'Yellow', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    for cnt in contours_orange:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(input_image_cpy, 'Orange', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    for cnt in contours_cyan:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(input_image_cpy, 'Cyan', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    for cnt in contours_magenta:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(input_image_cpy, 'Magenta', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    for cnt in contours_grey:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (128, 128, 128), 2)
        cv2.putText(input_image_cpy, 'Grey', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)

    for cnt in contours_white:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(input_image_cpy, 'White', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # display the input image with contours
    cv2.imshow('Color Recognition Output', input_image_cpy)
 
    # Close video window by pressing 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()