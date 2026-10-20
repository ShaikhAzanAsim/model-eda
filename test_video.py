import os
import cv2
import numpy as np

# Create Dummy videos
os.makedirs('video_dataset/cats', exist_ok=True)
os.makedirs('video_dataset/dogs', exist_ok=True)

print("Creating sample MP4s...")
out1 = cv2.VideoWriter('video_dataset/cats/cat.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
for i in range(120):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 100 + i*2), 50, (0, 0, 255), -1)
    out1.write(img)
out1.release()

out2 = cv2.VideoWriter('video_dataset/dogs/dog.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512, 512))
for i in range(120):
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.rectangle(img, (100 + i*2, 256), (200 + i*2, 356), (255, 0, 0), -1)
    out2.write(img)
out2.release()

print("MP4s generated. Running core logic...")

# Import modular pipeline without Qt GUI
import sys
sys.path.insert(0, 'Pipeline')
from dataset_detector import DatasetDetector
from annotation_handlers import AnnotationHandlerFactory
from video_generator import VideoGenerator

detector = DatasetDetector()
info = detector.detect_dataset('video_dataset')

print(f"Detected: {info.dataset_type} with {len(info.image_paths)} medias")

handler = AnnotationHandlerFactory().get_handler(info.dataset_type, info)
gen = VideoGenerator(fps=30)

all_samples = []
for p in info.image_paths:
    print(f"Annotating {p}...")
    sample_frames = handler.annotate_image(p, 2)
    print(f"  Got {len(sample_frames)} frames extracted.")
    all_samples.append(sample_frames)

# Duplicate randomly for grid 2x2 = 4 blocks
print("Assembling 2x2 grid via VideoGenerator...")
grid_samples = [all_samples[0], all_samples[1], all_samples[0], all_samples[1]]
gen.create_grid_video([grid_samples], 2, 'video_dataset/final_test_stitched_video.mp4')
print("Complete.")
