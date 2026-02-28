import cv2
import numpy as np
orig = cv2.imread('data/preprocessed_cropped/Scenario-A/Brazilian/track_00001/lr-001.png')
sr = cv2.imread('output/inference/sr_result/sr_lr-001.png')
print(f"Original shape: {orig.shape}, min: {orig.min()}, max: {orig.max()}")
print(f"SR shape: {sr.shape}, min: {sr.min()}, max: {sr.max()}")
