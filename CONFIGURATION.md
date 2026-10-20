# Configuration Reference

This document lists all tunable parameters in the Dataset Visualizer system.

## PyQt Application (`Pipeline/pyqt.py`)

### VisualizationWorker Parameters
```python
class VisualizationWorker(QThread):
    # Adjustable in __init__:
    self.dataset_path = dataset_path      # User input, path to dataset
    self.n_samples = n_samples             # User input, 1-1000
    self.grid_size = grid_size             # User input, 2-6
```

### Main Application Window
```python
class DatasetVisualizer(QWidget):
    # Window geometry
    self.setGeometry(100, 100, 700, 400)  # x, y, width, height
    
    # Spinbox constraints
    self.sample_spinbox.setMinimum(1)      # Min samples
    self.sample_spinbox.setMaximum(1000)   # Max samples
    self.sample_spinbox.setValue(10)       # Default value
    
    # Default grid size
    self.grid_combobox.setCurrentText("3x3")  # Default grid
```

---

## Dataset Detection (`Pipeline/dataset_detector.py`)

### Scan Parameters
```python
class DatasetDetector:
    # Image file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'}
    
    # Directory scan depth (change in _scan_directory)
    max_depth = 3                          # How deep to scan directories
    
    # Segmentation detection threshold
    match_ratio > 0.3                      # % of images with matching masks
```

### Format Detection Priority
```python
# In _detect_dataset_type():
priority = [
    'segmentation_mask',    # Highest priority
    'pascal_voc',
    'coco',
    'yolo',
    'labelme',
    'custom'                # Lowest priority
]
```

### Supported Annotation Extensions
```python
ANNOTATION_EXTENSIONS = {
    '.json': ['labelme', 'coco', 'custom'],
    '.xml': ['pascal_voc', 'vott'],
    '.txt': ['yolo', 'darknet'],
    '.mat': ['matlab'],
    '.png': ['segmentation_mask'],
    '.npy': ['numpy'],
}
```

---

## Annotation Handlers (`Pipeline/annotation_handlers.py`)

### ClassificationHandler
```python
class ClassificationHandler(AnnotationHandler):
    # Output image size
    size = (512, 512)
    
    # Text label position and format
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)  # White text
    thickness = 2
    position = (10, 30)
```

### SegmentationHandler
```python
class SegmentationHandler(AnnotationHandler):
    # Output image size
    size = (512, 512)
    
    # Overlay blending
    img_weight = 0.7         # Original image contribution
    mask_weight = 0.3        # Mask overlay contribution
    
    # Colormap for masks
    colormap = cv2.COLORMAP_JET  # Can change to COLORMAP_HOT, etc.
```

### DetectionHandler
```python
class DetectionHandler(AnnotationHandler):
    # Output image size
    size = (512, 512)
    
    # Bounding box drawing
    color = (0, 255, 0)      # Green boxes
    thickness = 2            # Box border width
    
    # Label text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
```

### PoseHandler
```python
class PoseHandler(AnnotationHandler):
    # Output image size
    size = (512, 512)
    
    # Keypoint drawing
    radius = 5               # Point circle radius
    
    # Keypoint confidence threshold
    confidence_threshold = 0.5  # Skip points below this confidence
    
    # Keypoint colors per person
    colors = [
        (0, 255, 0),         # Green
        (255, 0, 0),         # Blue
        (0, 0, 255),         # Red
        (255, 255, 0),       # Cyan
    ]
```

---

## Video Generator (`Pipeline/video_generator.py`)

### VideoGenerator Parameters
```python
class VideoGenerator:
    # Video encoding parameters
    fps = 30                 # Frames per second
    codec = 'mp4v'          # Video codec (H.264 compatible)
    
    # Sample display duration
    # In VisualizationWorker:
    frames.extend([frame] * 60)  # 60 frames at 30fps = 2 seconds per sample
    # Change 60 to show longer (e.g., 90 for 3 seconds)
```

### Adjustable Codecs
```python
# In create_video():
fourcc = cv2.VideoWriter_fourcc(*self.codec)

# Available options:
'mp4v'  # H.264 (current, recommended)
'MJPG'  # Motion JPEG
'DIVX'  # MPEG-4
'H264'  # Alternative H.264
'WMV1'  # Windows Media Video 1
```

---

## Performance Tuning

### For Large Datasets
```python
# Increase scan depth for nested directories
max_depth = 5              # Instead of default 3

# Reduce memory usage with smaller images
size = (256, 256)          # Instead of (512, 512)
```

### For Video Quality
```python
# Lower FPS for smoother playback on slow systems
fps = 24                   # Instead of default 30
```

### For Faster Processing
```python
# Reduce display time per sample
frames.extend([frame] * 30)   # 1 second instead of 2 seconds

# Reduce grid size (uses smaller, faster images)
grid_combobox options: "2x2" # Fastest

# Reduce number of samples
sample_spinbox minimum: 1
```

---

## Custom Handler Template

To add support for new dataset types, create a handler following this template:

```python
from annotation_handlers import AnnotationHandler
import cv2

class MyDatasetHandler(AnnotationHandler):
    """Handles MyDataset type annotations"""
    
    # Configuration options
    OUTPUT_SIZE = (512, 512)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    TEXT_COLOR = (0, 255, 0)  # BGR format
    
    def annotate_image(self, image_path: str, grid_cells: int):
        """
        Main annotation method
        
        Args:
            image_path: Path to image file
            grid_cells: Grid size (for reference)
        
        Returns:
            Annotated image as numpy array (BGR, uint8)
        """
        # Load image
        img = self._load_image(image_path).copy()
        h, w = img.shape[:2]
        
        # Find annotations
        ann_path = self._find_annotation_for_image(image_path, '.json')
        
        if ann_path:
            # Load and parse annotations
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
            
            # Draw annotations on image
            self._draw_annotations(img, annotations)
        
        # Add label
        img = self._add_label_text(img, "MyDataset")
        
        # Resize to standard size
        img = cv2.resize(img, self.OUTPUT_SIZE)
        
        return img
    
    def _draw_annotations(self, img, annotations):
        """Your custom drawing logic here"""
        # Example: Draw boxes
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Register handler
from annotation_handlers import AnnotationHandlerFactory
factory = AnnotationHandlerFactory()
factory.register_handler('mydataset', MyDatasetHandler)
```

---

## Environment Variables

None currently used. All configuration via code parameters or GUI inputs.

To add environment variable support:
```python
import os

# In dataset_detector.py
MAX_DEPTH = int(os.getenv('VISUALIZER_MAX_DEPTH', '3'))

# In video_generator.py  
DEFAULT_FPS = int(os.getenv('VISUALIZER_FPS', '30'))
```

---

## Configuration Files

No configuration files are used by default. To add config file support:

```yaml
# config.yaml (optional)
dataset_detection:
  max_depth: 3
  image_extensions: [jpg, png, bmp]
  segmentation_match_ratio: 0.3

video_generation:
  fps: 30
  codec: mp4v
  default_sample_duration_frames: 60

handlers:
  classification:
    output_size: [512, 512]
  segmentation:
    image_weight: 0.7
    mask_weight: 0.3
  detection:
    box_color: [0, 255, 0]
    box_thickness: 2
```

---

## Debugging Configuration

To enable verbose logging:

```python
# In pyqt.py, VisualizationWorker.run():
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Then use:
logger.debug(f"Detected {len(dataset_info.images)} images")
```

---

## Performance Benchmarks

On typical hardware (4GB RAM, SSD):

| Configuration | Time | Memory |
|--------------|------|--------|
| 10 samples, 3x3 grid | ~30 seconds | ~500 MB |
| 50 samples, 3x3 grid | ~120 seconds | ~1 GB |
| 10 samples, 6x6 grid | ~60 seconds | ~1.5 GB |

Video output: ~5-10 MB per minute of video

---

## Recommended Settings

### Quick Testing
```
Samples: 5
Grid Size: 2x2
Duration: 1 second per sample (frames *= 30)
```

### Full Exploration  
```
Samples: 20
Grid Size: 3x3
Duration: 2 seconds per sample (frames *= 60)
```

### Presentation
```
Samples: 50
Grid Size: 4x4
Duration: 3 seconds per sample (frames *= 90)
```

### Large Datasets
```
Samples: 100
Grid Size: 5x5
Duration: 2 seconds per sample (frames *= 60)
FPS: 24 (slightly lower for smoother playback)
```

