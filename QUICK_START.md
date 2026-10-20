# Quick Start Guide - Image Dataset Visualizer

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python Pipeline/pyqt.py
```

### 3. Use the GUI
1. **Browse** to your image dataset folder
2. Enter **sample count** (e.g., 10 for 10 random images)
3. Select **grid size** (2x2, 3x3, 4x4, etc.)
4. Click **Generate Slideshow**
5. Wait for processing
6. Video saved as `slideshow_video.mp4` in your dataset folder

---

## Supported Dataset Types

### ✅ Classification
Folder structure with class labels as subdirectories
```
dataset/
├── cats/
│   ├── image1.jpg
│   └── image2.jpg
└── dogs/
    ├── image1.jpg
    └── image2.jpg
```

### ✅ Segmentation
Images with corresponding mask files
```
dataset/
├── train_images/
│   ├── img001.jpg
│   └── img002.jpg
└── train_masks/
    ├── img001.png (grayscale mask)
    └── img002.png
```

### ✅ Object Detection
Images with annotation files (COCO, YOLO, or LabelMe)
```
dataset/
├── images/
│   ├── img1.jpg
│   └── img2.jpg
└── annotations/
    ├── img1.json (COCO/LabelMe format)
    ├── img1.txt  (YOLO format)
    └── img2.xml  (Pascal VOC format)
```

### ✅ Pose Estimation
Images with keypoint annotations
```
dataset/
├── images/
│   └── person1.jpg
└── keypoints/
    └── person1.json (COCO keypoints format)
```

---

## Features

| Feature | Details |
|---------|---------|
| **Auto-Detection** | Automatically identifies dataset type |
| **Directory Flexibility** | Works with any folder structure |
| **Random Sampling** | Select N random samples (1-1000) |
| **Grid Layout** | Display in 2x2, 3x3, 4x4, 5x5, or 6x6 grids |
| **Smart Annotation** | Shows appropriate annotations per dataset type |
| **Video Generation** | Creates MP4 slideshow (2+ sec per sample) |
| **Threaded Processing** | Non-blocking UI during processing |
| **Format Support** | COCO, YOLO, Pascal VOC, LabelMe, Custom |

---

## Common Workflows

### Workflow 1: Explore Your Dataset
```
1. Open app → Browse dataset
2. Set N=10 samples, Grid=3x3
3. Click Generate
4. Review slideshow video
```

### Workflow 2: Verify Annotations
```
1. Directory detected ✓
2. Type detected as "Detection" ✓
3. Bounding boxes drawn correctly ✓
4. Watch video to verify annotations
```

### Workflow 3: Create Custom Presentation
```
1. Set N=50 samples
2. Choose Grid=4x4
3. Generate 50-frame slideshow
4. Share MP4 video
```

---

## Troubleshooting

### Issue: "No images found"
- ✓ Check file extensions (.jpg, .png, .bmp, .gif, .webp)
- ✓ Verify folder permissions
- ✓ Use absolute path, not relative

### Issue: "Dataset type detected as Unknown"
**Solution**: Add a custom handler
```python
# Edit annotation_handlers.py
class MyDatasetHandler(AnnotationHandler):
    def annotate_image(self, image_path, grid_cells):
        # Your logic here
        return annotated_image

# Register
factory = AnnotationHandlerFactory()
factory.register_handler('my_type', MyDatasetHandler)
```

### Issue: Video generation fails
- ✓ Check disk space
- ✓ Ensure write permissions to dataset folder
- ✓ Try smaller grid size

### Issue: Wrong annotations shown
- ✓ Verify annotation format (COCO/YOLO/XML/JSON)
- ✓ Check annotation filenames match image names
- ✓ Ensure annotations folder is accessible

---

## Programmatic Usage

For batch processing or integration:

```python
from Pipeline.dataset_detector import DatasetDetector
from Pipeline.annotation_handlers import AnnotationHandlerFactory

# Detect
detector = DatasetDetector()
dataset_info = detector.detect_dataset('/path/to/dataset')

# Process
factory = AnnotationHandlerFactory()
handler = factory.get_handler(dataset_info.dataset_type, dataset_info)

# Annotate one image
annotated = handler.annotate_image(dataset_info.image_paths[0], grid_size=3)
```

See `examples.py` for more advanced usage.

---

## Output

**Location**: `slideshow_video.mp4` in your dataset folder

**Contents**:
- N random samples from your dataset
- 2+ seconds per sample (60 frames at 30fps)
- Appropriate annotations for dataset type
- Ready to share or present

---

## Next Steps

- 📖 Read [README_VISUALIZER.md](README_VISUALIZER.md) for full documentation
- 💻 Check [examples.py](examples.py) for programmatic usage
- 🔧 Extend with custom handlers for specialized datasets
- 🚀 Integrate into your analysis pipeline

---

## Need Help?

- Check README_VISUALIZER.md for detailed documentation
- Review examples.py for code samples
- Verify dataset structure matches examples above
- Check that all files are accessible and readable

Happy analyzing! 🎉
