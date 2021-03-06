import time
import cv2
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

def main(args,image_path):
    net, refine_net = load_detector()
    model, converter = load_recognizer()

    image = loadImage(image_path)
    # image = resize_img(image,640)
    image = cv2.resize(image,(480,960))
    # t_start = time.time()
    bboxes, polys, score_text = extract_text_box(net, image, args.text_threshold,
                                                 args.link_threshold, args.low_text,
                                                 args.cuda, args.poly, refine_net)
    text = extract_text(model, converter, image, polys, image_path)
    print(text)

if __name__ == '__main__':
    args = get_config()
    img_path = "/home/son/Desktop/datn20201/resource/img/receipt_1007.jpg"
    main(args,img_path)
