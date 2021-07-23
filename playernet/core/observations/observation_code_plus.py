import pyautogui
import cv2

# Capture the screen
capture = pyautogui.screenshot()

# Save the image
capture.save("capture_screen.png")

# Capture only part of the screen
capture_region = pyautogui.screenshot(region=(0, 0, 1920, 1080))


img = cv2.imread('capture_screen.png', cv2.IMREAD_UNCHANGED)

print('Original Dimensions : ', img.shape)

width = 640
height = 360
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

print('Resized Dimensions : ', resized.shape)

# cv2.imshow("Resized image", resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Save the image
save_img = cv2.imwrite('resized_screen.png', resized)