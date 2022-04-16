import cv2
import numpy as np

def draw_bbox(images, bboxes, logits = None, mask = None, label = None, threshold = 0.5, mix_ratio = 0.5, method = cv2.INTER_LINEAR, prefix = "", postfix = "", color = None):
    images = np.array(images)
    bboxes = np.array(bboxes)
    for batch_index in range(len(images)):
        image = images[batch_index]
        bbox = bboxes[batch_index]
        for index in range(len(bbox)):
            rect = bbox[index]
            if 0 < np.max(rect):
                bbox_color = color
                if color is None:
                    bbox_color = np.random.random(size = 3)
                    if 1 < np.max(image):
                        bbox_color = np.round(bbox_color * 255)
                    bbox_color = tuple(bbox_color)
                h, w = image.shape[:2]
                if np.max(rect) <= 1:
                    rect *= [w, h, w, h]
                rect = tuple(rect.astype(int))
                size = int(max(h, w) / 500)
                cv2.rectangle(image, rect[:2], rect[-2:], bbox_color, size)
                
                if logits is not None:
                    logits_color = (1, 1, 1)
                    if 1 < np.max(image):
                        logits_color = (255, 255, 255)
                    prob = np.array(logits[batch_index][index])
                    if np.shape(prob)[-1] != 1:
                        prob_index = np.argmax(prob, axis = -1)
                        score = prob[prob_index]
                    else:
                        prob_index = prob[..., 0]
                        score = np.ones_like(prob_index, dtype = np.float32)
                    name = prob_index
                    if label is not None:
                        name = label[prob_index]
                    msg = "{0}{1}:{2:.2f}{3}".format(prefix, name, score, postfix)
                    font_size = max(h, w) / 1250
                    text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_size, size)[0]
                    font_pos = (rect[0], max(rect[1], text_size[1]))
                    cv2.rectangle(image, (font_pos[0], font_pos[1] - text_size[1]), (font_pos[0] + text_size[0], font_pos[1]), bbox_color, -1)
                    cv2.putText(image, msg, font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, logits_color, size)
                    
                if mask is not None:
                    m = np.array(mask[batch_index][index])
                    m = cv2.resize(m, (min(rect[2] + 1, w) - rect[0], min(rect[3] + 1, h) - rect[1]))
                    m = np.where(threshold <= m, 1., 0.)
                    m = np.tile(np.expand_dims(m, axis = -1), 3) * bbox_color
                    crop = image[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1]
                    image[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1] = np.where(0 < m, crop * (1 - mix_ratio) + m * mix_ratio, crop)
    return images