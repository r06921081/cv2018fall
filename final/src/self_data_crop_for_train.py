import cv2
import numpy as np
from util import *
path = "/media/cnrg-ntu2/HDD1TB/r07921052/cv_course/CV_finalproject/data/Synthetic/"

left_image = ["TL0.png","TL1.png","TL2.png", "TL3.png", "TL4.png", "TL5.png","TL6.png", "TL7.png", "TL8.png","TL9.png"]
right_image = ["TR0.png","TR1.png","TR2.png", "TR3.png", "TR4.png", "TR5.png","TR6.png", "TR7.png", "TR8.png","TR9.png"]
disp_image = ["TLD0.pfm","TLD1.pfm","TLD2.pfm","TLD3.pfm","TLD4.pfm","TLD5.pfm","TLD6.pfm","TLD7.pfm","TLD8.pfm","TLD9.pfm"]


t = []

for i in range(10):
    gt = readPFM("./data/Synthetic/TLD"+str(i)+".pfm")
    h, w = gt.shape
    
    #x = cv2.imread("TL"+str(i)+".pngsub.png", 0) #str(i)+"result.png"
    #x = cv2.imread("TL"+str(i)+".wmf.png", 0)
    #x = cv2.resize(x, (w,h))
    x = readPFM("TL"+str(i)+".pfm") #str(i)+"result.pfm" str(i)+"result_sub.pfm" "TL"+str(i)+"wmf.pfm"
    #x = readPFM("TL0.pfm")
    disp = x.flatten()


    gt = gt.flatten()
    err = cal_avgerr(gt, disp)
    t.append(err)
    print(err)

print("avg error: ", sum(t)/len(t))
