#!/usr/bin/env python
import numpy as np
import model
import cv2
import random

random.seed(1)

class BoundBox:
    def __init__(self, x, y, w, h, c = None, classes = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score
    def __str__(self):
        return "X:{} Y:{} W:{} H:{} C:{}".format(self.x, self.y, self.w, self.h, self.get_label())


def bbox_iou(box1, box2):
    x1_min  = box1.x - box1.w/2
    x1_max  = box1.x + box1.w/2
    y1_min  = box1.y - box1.h/2
    y1_max  = box1.y + box1.h/2
    
    x2_min  = box2.x - box2.w/2
    x2_max  = box2.x + box2.w/2
    y2_min  = box2.y - box2.h/2
    y2_max  = box2.y + box2.h/2
    
    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])    
    intersect = intersect_w * intersect_h    
    union = box1.w * box1.h + box2.w * box2.h - intersect
    
    return float(intersect) / union
    
def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def mergeset_images(src_image, x, y, dest_image):
    if x < 0:
        src_image = src_image[:,abs(x):]
        x = 0

    if y < 0:
        src_image = src_image[abs(y):,:]
        y = 0

    nh,nw = dest_image.shape[:2]
    oh,ow = src_image.shape[:2]

    aw = nw - x
    ah = nh - y

    if ow > aw:
        src_image = src_image[:,:aw]
        ow = aw

    if oh > ah:
        src_image = src_image[:ah,:]
        oh = ah

    dest_image[y:y+oh,x:x+ow] = src_image
    return dest_image

def scale_image(im_data, w, h):
    im_h, im_w, z = im_data.shape
    wp = w/float(im_w)
    hp = h/float(im_h)
    if wp < hp:
        nw = w
        nh = (im_h * nw) /im_w
    else:
        nh = h
        nw = (im_w * nh) /im_h

    return (nw,nh)

def letter(im_data, w, h, color=127):
    n_shape = scale_image(im_data, w,h)
    im_dest = (np.ones((h,w,3))*color).astype(np.uint8)
    im_data = cv2.resize(im_data, (n_shape[0], n_shape[1]), interpolation=cv2.INTER_NEAREST)
    x = (im_dest.shape[1] - im_data.shape[1]) / 2
    y = (im_dest.shape[0] - im_data.shape[0]) / 2
    im_dest = mergeset_images(im_data, x, y, im_dest)
    return im_dest

def unletter_boxes(boxes, out_img, input_w, input_h):
    dim_change = scale_image(out_img, input_w, input_h)
    w_change = input_w - dim_change[0]
    h_change = input_h - dim_change[1]

    for box in boxes:
        x = (box.x * input_w) - w_change / 2.0
        y = (box.y * input_h) - h_change / 2.0
        w = box.w * input_w
        h = box.h * input_h

        box.x = x/(input_w - w_change)
        box.w = w/(input_w - w_change)
        box.y = y/(input_h - h_change)
        box.h = h/(input_h - h_change)

if __name__ == "__main__":
    ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
    # ANCHORS = '1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.00711'
    CLASSES = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

    ANCHOR_VALUES = np.reshape([float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')], [5,2])    
    box_colors = [(int(random.random()*255),int(random.random()*255),int(random.random()*255)) for i in range(20)]
    
    fe = model.FeatureExtractor()
    net = fe.yolo_convolutional_net()
    net.summary()
    net.load_weights("yolo-voc.1.0.h5")

    cap = cv2.VideoCapture("pedestrian.mp4")
    while True:
        tmp,im_data = cap.read()
        im_h, im_w = im_data.shape[:2]
        im_out = im_data.copy()
        im_data = letter(im_data, 416, 416)                    
        im_data = im_data[:,:,::-1]
        im_data = im_data.astype(np.float32).reshape((1,416, 416,3)) 
        im_data /= 255.0        
        fake_boxes = np.zeros((1, 1, 1, 1, 15, 4))
        fake_anchors = np.zeros((1, 13, 13, 5, 1))          
        preds = net.predict([im_data, fake_boxes, fake_anchors], batch_size=1, verbose=0)
        preds = preds[0].reshape((13, 13, 5, 25))                
        dim, detectors, n_classes = (13, 5, 20)
        boxes = []
        for row in range(dim):
            for col in range(dim):
                for n in range(detectors):
                    x,y = (sigmoid(preds[row,col,n,:2])+[col,row])/dim
                    w,h = (np.exp(preds[row,col,n,2:4]) * ANCHOR_VALUES[n])/dim
                    scale = sigmoid(preds[row,col,n,4])
                    classes_scores = softmax(preds[row,col,n,5:])*scale

                    if np.sum(classes_scores * (classes_scores>.3)) > 0:
                        box = BoundBox(x, y, w, h, scale, classes_scores)
                        boxes += [box]

        for c in range(20):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))
            for i in xrange(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in xrange(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if bbox_iou(boxes[index_i], boxes[index_j]) >= .3:
                            boxes[index_j].classes[c] = 0
                            
        boxes = [box for box in boxes if box.get_score() > .3]
        unletter_boxes(boxes, im_out, 416, 416)
        im_h, im_w = im_out.shape[:2]
                
        for box in boxes:            
            xl_f = (box.x - box.w / 2.0) * im_w
            yt_f = (box.y - box.h / 2.0) * im_h
            xr_f = (box.x + box.w / 2.0) * im_w
            yb_f = (box.y + box.h / 2.0) * im_h

            xl_f = 1.0 if xl_f < 1.0 else xl_f
            yt_f = 1.0 if yt_f < 1.0 else yt_f
            xr_f = im_w if xr_f > im_w else xr_f
            yb_f = im_h if yb_f > im_h else yb_f

            xl = int(xl_f)
            yt = int(yt_f)
            xr = int(xr_f)
            yb = int(yb_f)

            bottomLeftCornerOfText = (xl,yt-8)
            cv2.putText(im_out, str(box.get_score())[:4] + " " + CLASSES[box.get_label()], bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 6)
            cv2.putText(im_out, str(box.get_score())[:4] + " " + CLASSES[box.get_label()], bottomLeftCornerOfText, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            cv2.rectangle(im_out, (xl, yt), (xr, yb), box_colors[box.get_label()], 3)
            cv2.rectangle(im_out, ((xl+xr)/2-1, (yt+yb)/2-1), ((xl+xr)/2+1, (yt+yb)/2+1), (0,0,255), 3)

        cv2.imshow('frame',im_out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
