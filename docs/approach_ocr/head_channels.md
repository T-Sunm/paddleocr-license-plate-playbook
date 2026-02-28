# Feature 2 — Dynamic Recognition Head Channels

### The Problem

License plates use a restricted character set — typically **38 characters** (digits + uppercase letters + blank). PaddleOCR's default `MultiHead` (used for NRTR + CTC + SAR heads) was hardcoded to the standard 97-char English dictionary.

When you pass a custom `character_dict_path`, the label decoder produces the correct number of classes, but the head's `out_channels` is still hardcoded to 97. This causes a **shape mismatch** at the loss layer the moment training begins.

### The Fix

**File modified:** `PaddleOCR/ppocr/modeling/heads/rec_multi_head.py`

The `MultiHead.__init__` is modified to read output sizes from the `out_channels_list` dictionary (which PaddleOCR already populates from the label decoder) instead of hardcoded integers:

```python
# rec_multi_head.py — MultiHead.__init__

# Before (hardcoded):
# self.ctc_head = CTCHead(in_channels=..., out_channels=97)

# After (dynamic via out_channels_list dict):
for idx, head_name in enumerate(self.head_list):
    name = list(head_name)[0]

    if name == "CTCHead":
        self.ctc_head = CTCHead(
            in_channels=self.ctc_encoder.out_channels,
            out_channels=out_channels_list["CTCLabelDecode"],  # <-- dynamic
        )
    elif name == "NRTRHead":
        self.gtc_head = Transformer(
            ...
            out_channels=out_channels_list["NRTRLabelDecode"],  # <-- dynamic
        )
    elif name == "SARHead":
        self.sar_head = SARHead(
            in_channels=in_channels,
            out_channels=out_channels_list["SARLabelDecode"],   # <-- dynamic
        )
```

All three head types (`CTCLabelDecode`, `NRTRLabelDecode`, `SARLabelDecode`) are covered. PaddleOCR automatically fills this dict when you set `character_dict_path` in the Global config.

### Usage in Config

```yaml
# configs/rec_v5_template.yaml
Global:
  character_dict_path: data/plate_char_dict.txt   # 38 chars for plates

Head:
  name: MultiHead
  head_list:
    - CTCHead:
        Neck:
          name: svtr
    - NRTRHead:
        nrtr_dim: 256
        max_text_length: 10
```

No additional changes needed — once `character_dict_path` is set, the dict is populated automatically and `MultiHead` reads the correct sizes.

---

