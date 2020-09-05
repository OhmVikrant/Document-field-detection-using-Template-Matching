import cv2
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline


# whole image to search

doc = cv2.imread('./input/constitution.jpg')
doc = cv2.cvtColor(doc, cv2.COLOR_BGR2RGB)


# The Template to Match

temp = cv2.imread('./input/constitution_template.jpg')
temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)



# Apply template Matching with the method
match = cv2.matchTemplate(doc, temp, eval('cv2.TM_CCOEFF'))
    
# Grab the Max and Min values, plus their locations
_, _, _, maxLoc = cv2.minMaxLoc(match)

top_left_corner = maxLoc
bottom_right_corner = (top_left_corner[0] + width, top_left_corner[1] + height)

# Draw the Red Rectangle
cv2.rectangle(doc,top_left_corner, bottom_right_corner, 255, 10)

plt.figure(figsize = (20,20))
plt.imshow(doc)
plt.title('Detected Point')
plt.show()



