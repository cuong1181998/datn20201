import time
import cv2
import os
import numpy as np
 
try :
   from text_detect import load_detector, extract_text_box, resize_img
   from text_recognize import load_recognizer, extract_text
   from detection.imgproc import loadImage
   from opt import get_config
except ImportError:
   from src.text_detect import load_detector, extract_text_box, resize_img
   from src.text_recognize import load_recognizer, extract_text
   from src.detection.imgproc import loadImage
   from src.opt import get_config
 
def get_images(folder):
   files = []
   exts = ['jpg', 'png', 'jpeg', 'JPG']
   for parent, dirnames, filenames in os.walk(folder):
       for filename in filenames:
           for ext in exts:
               if filename.endswith(ext):
                   files.append(os.path.join(parent, filename))
                   break
   print('Find {} images'.format(len(files)))
   return files
  
def main(args,folder):
   net, refine_net = load_detector()
   model, converter = load_recognizer()
 
   im_fn_list = get_images(folder)
   for im_fn in im_fn_list:
     im = loadImage(im_fn)
     image = cv2.resize(im,(480,960))
     bboxes, polys, score_text = extract_text_box(net, image, args.text_threshold,
                                                 args.link_threshold, args.low_text,
                                                 args.cuda, args.poly, refine_net)
     text = extract_text(model, converter, image, polys, im_fn)
     for raw in text:
       box = np.array([int(raw[1]), int(raw[2]), int(raw[3]), int(raw[4]), int(raw[5]), int(raw[6]), int(raw[7]), int(raw[8])])
       title = raw[9];
       cv2.polylines(image, [box.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)  
       cv2.putText(image, title, (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
     print(os.path.join('../result', os.path.basename(im_fn)))
     cv2.imwrite(os.path.join('../result', os.path.basename(im_fn)),image)
if __name__ == '__main__':
   args = get_config()
   main(args,args.img_path)