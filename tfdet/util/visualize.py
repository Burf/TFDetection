import cv2
import numpy as np

def draw_bbox(images, bboxes, logits = None, mask = None, label = None, threshold = 0.5, mix_ratio = 0.5, method = cv2.INTER_LINEAR, probability = True, prefix = "", postfix = "", color = None):
    batch = True
    if np.ndim(images) not in [1, 4]:
        batch = False
        images = [images]
        bboxes = [bboxes]
        if logits is not None:
            logits = [logits]
        if mask is not None:
            mask = [mask]

    result = []
    for batch_index in range(len(images)):
        image = np.array(images[batch_index])
        bbox = np.array(bboxes[batch_index])
        h, w = np.shape(image)[:2]
        size = int(max(h, w) / 500)
        font_size = max(h, w) / 1250
        normalize_flag = np.max(image) <= 1
        logits_color = (1, 1, 1) if normalize_flag else (255, 255, 255)
        
        valid_indices = np.where(0 < np.max(bbox, axis = -1))
        bbox = bbox[valid_indices]
        _mask = np.array(mask[batch_index])[valid_indices] if mask is not None else None
        if logits is not None:
            logit = np.array(logits[batch_index])[valid_indices]
            if np.shape(logit)[-1] != 1:
                logit_index = np.argmax(logit, axis = -1)
                score = np.max(logit, axis = -1)
            else:
                logit_index = logit[..., 0].astype(int)
                score = np.ones_like(logit_index, dtype = np.float32)
        
        for index, rect in enumerate(bbox):
            bbox_color = color
            if color is None:
                bbox_color = np.random.random(size = 3) if normalize_flag else np.random.randint(0, 256, size = 3)
            bbox_color = tuple(bbox_color)
            if np.max(rect) <= 1:
                rect = np.round(np.multiply(rect, [w, h, w, h]))
            rect = tuple(rect.astype(int))
            
            if logits is not None:
                name = label[logit_index[index]] if label is not None else logit_index[index]
                bbox_color = bbox_color[logit_index[index]] if np.ndim(bbox_color) == 2 else bbox_color
                msg = "{0}{1}".format(prefix, name)
                if probability:
                    msg = "{0}:{1:.2f}".format(msg, score[index])
                msg = "{0}{1}".format(msg, postfix)
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_size, size)[0]
                font_pos = (rect[0], max(rect[1], text_size[1]))
            
            cv2.rectangle(image, rect[:2], rect[-2:], bbox_color, size)
            if logits is not None:
                cv2.rectangle(image, (font_pos[0], font_pos[1] - text_size[1]), (font_pos[0] + text_size[0], font_pos[1]), bbox_color, -1)
                cv2.putText(image, msg, font_pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, logits_color, size)

            if mask is not None:
                m = _mask[index]
                m = cv2.resize(m, (min(rect[2] + 1, w) - rect[0], min(rect[3] + 1, h) - rect[1]), interpolation = method)
                m = np.where(threshold <= m, 1., 0.)
                m = np.tile(np.expand_dims(m, axis = -1), 3) * bbox_color
                crop = image[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1]
                image[rect[1]:rect[3] + 1, rect[0]:rect[2] + 1] = np.where(0 < m, crop * (1 - mix_ratio) + m * mix_ratio, crop)
        result.append(image)
    if not batch:# and len(images) == 1:
        result = result[0]
    else:
        result = np.array(result)
    return result
