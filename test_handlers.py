import sys
import cv2
import os

sys.path.insert(0, 'Pipeline')
from dataset_detector import DatasetDetector
from annotation_handlers import AnnotationHandlerFactory

datasets = ['imagenet', 'segment', 'minicoco']
factory = AnnotationHandlerFactory()
detector = DatasetDetector()

os.makedirs('test_output', exist_ok=True)

for ds in datasets:
    print(f"Testing {ds}...")
    info = detector.detect_dataset(ds)
    if not info:
        print(f"Failed to detect {ds}")
        continue
    
    print(f"  Type: {info.dataset_type}")
    print(f"  Classes: {len(info.class_names)}")
    
    handler = factory.get_handler(info.dataset_type, info)
    print(f"  Handler: {type(handler).__name__}")
    
    if info.image_paths:
        sample_img = info.image_paths[0]
        # Test cache population
        if getattr(info, 'annotation_format', '') == 'coco':
            cache_size = len(handler.coco_image_cache)
            print(f"  COCO Cache entries: {cache_size}")
            
        annotated = handler.annotate_image(sample_img, 1)
        if annotated is not None:
            out_file = f'test_output/{ds}_annotated.jpg'
            cv2.imwrite(out_file, annotated)
            print(f"  Saved {out_file}")
        else:
            print("  Failed to annotate.")
    else:
        print("  No images found.")
