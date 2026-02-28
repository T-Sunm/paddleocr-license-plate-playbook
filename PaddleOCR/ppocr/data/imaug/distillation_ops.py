import numpy as np
import cv2
from ppocr.data.imaug.rec_img_aug import RecResizeImg

class DecodeMultiImagePair(object):
    def __init__(self, img_mode="BGR", channel_first=False, pad_value=0, **kwargs):
        self.img_mode = img_mode
        self.channel_first = channel_first
        self.pad_value = pad_value
        self._fail_count = 0

    def _decode_and_stack(self, img_bytes_list):
        if not img_bytes_list:
            print(f"[DEBUG] Empty img_bytes_list")
            return None
        
        decoded = []
        for i, img_bytes in enumerate(img_bytes_list):
            img_np = np.frombuffer(img_bytes, dtype="uint8")
            flag = cv2.IMREAD_COLOR
            img = cv2.imdecode(img_np, flag)
            if img is None:
                print(f"[DEBUG] Failed to decode image {i}, bytes len: {len(img_bytes)}")
                return None
            if self.img_mode == "RGB":
                img = img[:, :, ::-1]
            decoded.append(img)
        
        hs, ws = [im.shape[0] for im in decoded], [im.shape[1] for im in decoded]
        H, W = max(hs), max(ws)
        padded = []
        for im in decoded:
            h, w = im.shape[:2]
            if h != H or w != W:
                im = cv2.copyMakeBorder(im, 0, H - h, 0, W - w, 
                    cv2.BORDER_CONSTANT, value=(self.pad_value,)*3)
            padded.append(im)
        return np.stack(padded, axis=0)

    def __call__(self, data):
        img_data = data.get("image")
        if isinstance(img_data, dict) and "lr" in img_data:
            lr_stack = self._decode_and_stack(img_data["lr"])
            if lr_stack is None:
                return None
            data["image"] = lr_stack
            
            if img_data.get("hr") is not None:
                hr_stack = self._decode_and_stack(img_data["hr"])
                if hr_stack is None:
                    return None
                data["image_hr"] = hr_stack
            else:
                data["image_hr"] = None
            return data
        else:
            self._fail_count += 1
            if self._fail_count <= 3:
                print(f"[DEBUG] img_data is not dict with lr/hr, type: {type(img_data)}, path: {data.get('img_path', 'unknown')}")
            return None

class RecResizeImgPair(RecResizeImg):
    def __call__(self, data):
        data = super().__call__(data)
        if "image_hr" in data:
            temp_img = data["image"]
            data["image"] = data["image_hr"]
            data = super().__call__(data)
            data["image_hr"] = data["image"]
            data["image"] = temp_img
        return data
