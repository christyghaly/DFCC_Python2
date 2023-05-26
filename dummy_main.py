# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:05:06 2023

@author: hp
"""




import cv2




import radavg


img = cv2.imread('mask_beforescaling.png')
print(type(img))
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
radavg.radavg(gray,0.088)

# mask_new_size_x = (gray.shape[0]*2)-1
# mask_new_size_y = (gray.shape[1]*2)-1
# temp_mask = im.fromarray(gray)
# new_resized_mask = temp_mask.resize((mask_new_size_x,mask_new_size_y),im.BICUBIC)

# new_resized_mask.save('dummy.png')

# img2 = cv2.imread('dummy.png')

# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# #maskc, radius = innerCircle.innerCircle(gray)

# #print("radius", radius)
# cv2.imshow('contours',gray2)
# cv2.waitKey(0)  
# cv2.destroyAllWindows() 