import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

## Make canvas and set the color
img = np.zeros((200,400,3),np.uint8)
b,g,r,a = 0,255,0,0

# Use arial.ttf to write vietnamese
fontpath = "./arial.ttf" 
font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(img)
draw = ImageDraw.Draw(img_pil)
draw.text((50, 80),  "hehe làm được òi", font = font, fill = (b, g, r, a))
img = np.array(img_pil)


## Display 
# cv2.imshow("res", img);cv2.waitKey();cv2.destroyAllWindows()
#cv2.imwrite("res.png", img)