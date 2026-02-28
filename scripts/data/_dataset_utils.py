import logging

def get_clean_images(images):
    """Lọc ảnh gốc (không augmented), sorted by name."""
    return sorted([img for img in set(images) if "_aug_" not in img.name], key=lambda x: x.name)

def get_aug_groups(images):
    """Gom nhóm file augmentation"""
    groups = {}
    for img in set(images):
        if "_aug_" in img.name:
            suffix = img.name.split("_aug_")[1].split(".")[0]
            groups.setdefault(suffix, []).append(img)
    return {k: sorted(v, key=lambda x: x.name) for k, v in groups.items()}

def write_sequences(f, images, label, seq_len):
    """Ghi các sequence từ list images."""
    count = 0
    for i in range(0, len(images) - seq_len + 1, seq_len):
        seq = images[i:i+seq_len]
        paths = ",".join([str(p.resolve()) for p in seq])
        f.write(f"{paths}\t{label}\n")
        count += 1
    return count
