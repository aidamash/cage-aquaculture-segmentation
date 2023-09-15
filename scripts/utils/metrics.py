from collections import defaultdict
import cv2


def compute_metrics(mask, pred, c_i, min_area_threshold=100):
    mask_cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_cnts = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_cnts = mask_cnts[0] if len(mask_cnts) == 2 else mask_cnts[1]
    pred_cnts = pred_cnts[0] if len(pred_cnts) == 2 else pred_cnts[1]

    m_overlaps, p_overlaps, iou_score = defaultdict(list), defaultdict(list), defaultdict(list)
    for m_i, m_cnt in enumerate(mask_cnts):
        m_x, m_y, m_w, m_h = cv2.boundingRect(m_cnt)
        m_area = (mask[m_y:m_y + m_h, m_x:m_x + m_w] > 0).sum()
        m_overlaps[(c_i, m_i)].append(m_area)

        for p_i, p_cnt in enumerate(pred_cnts):
            p_x, p_y, p_w, p_h = cv2.boundingRect(p_cnt)
            p_area = (pred[p_y:p_y + p_h, p_x:p_x + p_w] > 0).sum()
            if p_area < min_area_threshold:
                continue

            if (c_i, p_i) not in p_overlaps:
                p_overlaps[(c_i, p_i)].append(p_area)

            # check if we've an overlap
            x_left = max(m_x, p_x)
            y_top = max(m_y, p_y)
            x_right = min(m_x + m_w, p_x + p_w)
            y_bottom = min(m_y + m_h, p_y + p_h)

            if x_right < x_left or y_bottom < y_top:  # no overlap
                continue

            m_overlaps[(c_i, m_i)].append(p_area)
            p_overlaps[(c_i, p_i)].append(m_area)

            # compute intersection
            int_mask = mask[y_top:y_bottom, x_left:x_right] * pred[y_top:y_bottom, x_left:x_right]
            int_area = (int_mask > 0).sum()
            iou_score[(c_i, m_i, p_i)] = int_area / (m_area + p_area - int_area)

    return m_overlaps, p_overlaps, iou_score
