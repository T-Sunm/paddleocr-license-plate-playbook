import numpy as np
import cv2
import os
from ppocr.data.simple_dataset import MultiScaleDataSet
from ppocr.data.imaug import transform


class DistillationDataSet(MultiScaleDataSet):
    MAX_RETRY = 5

    def __getitem__(self, properties):
        if isinstance(properties, (int, np.integer)):
            return self._getitem_eval(properties)
        return self._getitem_train(properties, retry=0)

    def _parse_lr_hr(self, file_name):
        """Parse 'lr1,lr2,...|hr1,hr2,...' format."""
        if "|" in file_name:
            lr_s, hr_s = file_name.split("|", 1)
            lr = [p.strip() for p in lr_s.split(",") if p.strip()]
            hr = [p.strip() for p in hr_s.split(",") if p.strip()]
            return {"lr": lr, "hr": hr}
        else:
            lr = [p.strip() for p in file_name.split(",") if p.strip()]
            return {"lr": lr, "hr": None}

    def _resolve_path(self, p):
        """Resolve path: use as-is if absolute, else join with data_dir."""
        return p if os.path.isabs(p) else os.path.join(self.data_dir, p)

    def _load_image(self, info):
        """Load LR and HR image bytes."""
        lr = [open(self._resolve_path(p), "rb").read() for p in info["lr"]]
        hr = None
        if info["hr"] is not None:
            hr = [open(self._resolve_path(p), "rb").read() for p in info["hr"]]
        return {"lr": lr, "hr": hr}

    def _get_data(self, file_idx):
        """Parse line and load images."""
        data_line = self.data_lines[file_idx].decode("utf-8").strip("\n")
        parts = data_line.split()
        file_name, label = parts[0], parts[-1]
        info = self._parse_lr_hr(file_name)
        img = self._load_image(info)
        return {"image": img, "label": label, "img_path": info["lr"][0], "ext_data": self.get_ext_data()}

    def _getitem_train(self, properties, retry=0):
        """Train mode with MultiScaleSampler."""
        img_height = properties[1]
        idx = properties[2] if len(properties) == 4 else properties[0]
        wh_ratio = properties[3] if len(properties) == 4 else None
        
        if wh_ratio is not None:
            img_width = img_height * max(1, int(round(wh_ratio)))
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = self.data_idx_order_list[idx]
            img_width = properties[0]

        try:
            data = self._get_data(file_idx)
            outs = transform(data, self.ops[:-1])
            if outs is None or "image_hr" not in outs:
                raise ValueError("Missing image_hr")
            outs = self._resize_both(outs, img_width, img_height)
            return transform(outs, self.ops[-1:])
        except Exception as e:
            if retry < self.MAX_RETRY:
                return self._getitem_train([img_width, img_height, (idx + 1) % len(self), None], retry + 1)
            raise RuntimeError(f"Failed after {self.MAX_RETRY} retries: {e}")

    def _getitem_eval(self, idx, retry=0):
        """Eval mode."""
        file_idx = self.data_idx_order_list[idx]
        try:
            data = self._get_data(file_idx)
            outs = transform(data, self.ops[:-1])
            if outs is not None and "image_hr" in outs and outs["image_hr"] is not None:
                self._resize_pair(outs)
            return transform(outs, self.ops[-1:])
        except Exception as e:
            if retry < self.MAX_RETRY:
                return self._getitem_eval((idx + 1) % len(self), retry + 1)
            raise RuntimeError(f"Failed after {self.MAX_RETRY} retries: {e}")

    def _resize_both(self, data, imgW, imgH):
        """Resize both LR and HR to same size for training."""
        lr, hr = data["image"], data["image_hr"]
        
        # Resize LR with padding (preserves aspect ratio)
        data["image"] = lr
        data = super().resize_norm_img(data, imgW, imgH, padding=True)
        lr_out = data["image"]
        
        # Resize HR to exact same size (no padding, force resize)
        data["image"] = hr
        data = super().resize_norm_img(data, imgW, imgH, padding=False)
        
        data["image"], data["image_hr"] = lr_out, data["image"]
        return data

    def _resize_pair(self, data):
        """Resize HR to match LR shape for eval."""
        if "image_hr" not in data or data["image_hr"] is None:
            return data
        img, hr = data["image"], data["image_hr"]
        if len(img.shape) == 4:
            T, C, H, W = img.shape
            frames = [cv2.resize(hr[t].transpose(1,2,0), (W,H)).transpose(2,0,1) for t in range(min(hr.shape[0], T))]
            frames += [frames[-1]] * (T - len(frames))
            data["image_hr"] = np.stack(frames[:T], axis=0).astype(np.float32)
        else:
            C, H, W = img.shape
            data["image_hr"] = cv2.resize(hr.transpose(1,2,0), (W,H)).transpose(2,0,1).astype(np.float32)
