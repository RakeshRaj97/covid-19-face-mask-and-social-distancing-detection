import os
import cv2
import glob
import statistics

temp = ['with_mask','without_mask']
height = []
width = []
for d in temp:
	img_dir = f"./{d}"
	img_path = os.path.join(img_dir,'*g')
	images = glob.glob(img_path)	
	for f in images:
		img = cv2.imread(f)
		h, w = img.shape[:2]
		height.append(h)
		width.append(w)

print(statistics.mean(height))
print(statistics.mean(width))
