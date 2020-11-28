import time
import cv2
import os
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
      im = cv2.imread(im_fn)[:, :, ::-1]
      image = cv2.resize(im,(480,960))
      t_start_1 = time.time()
      image_path = folder + os.path.splitext(os.path.basename(im_fn))[0] + os.path.splitext(os.path.basename(im_fn))[1]
      print(image_path)
      t_start_2 = time.time()
      bboxes, polys, score_text = extract_text_box(net, image, args.text_threshold,
                                                  args.link_threshold, args.low_text,
                                                  args.cuda, args.poly, refine_net)
      text = extract_text(model, converter, image, polys, image_path)
      print('ALL: ' + time.time() - t_start_1)
      print('REG: ' + time.time() - t_start_2)
if __name__ == '__main__':
    args = get_config()
    main(args,args.img_path)
