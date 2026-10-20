# Image Dataset Visualizer & Analyzer

A PyQt5-based application for exploring and analyzing image datasets with automatic detection of dataset types and intelligent visualization.

## Features

✨ **Automatic Dataset Detection**
- Detects dataset type: Classification, Segmentation, Object Detection, Pose Estimation
- Analyzes directory structure (co-located or separate image/annotation folders)
- Supports multiple annotation formats: COCO, Pascal VOC, YOLO, LabelMe, Custom JSON

🎨 **Smart Visualization**
- Displays random samples from the dataset
- Annotates images based on dataset type
- Shows overlays for segmentation masks
- Draws bounding boxes for detection
- Visualizes keypoints for pose estimation

🎬 **Video Generation**
- Creates slideshow video of selected samples
- Each sample displayed for 2+ seconds
- Generates MP4 video with configurable grid layout
- Saves video in the dataset directory

🔧 **Generic & Extensible**
- Factory pattern for easy addition of new annotation handlers
- Auto-detection adapts to various dataset structures
- Supports custom annotation formats

## Installation

1. **Install Dependencies**
```bash
cd eda_pipeline
pip install -r requirements.txt
```

2. **Run the Application**
```bash
python Pipeline/pyqt.py
```

## Usage

### Basic Workflow

1. **Launch Application**
   - Run `python Pipeline/pyqt.py`
   - A PyQt5 window will appear

2. **Select Dataset Directory**
   - Click "Browse" or enter path directly
   - Application scans for images and annotations

3. **Configure Parameters**
   - **Number of Samples (N)**: How many random samples to select (1-1000)
   - **Grid Size**: Layout dimension (2x2, 3x3, 4x4, 5x5, 6x6)

4. **Generate Slideshow**
   - Click "Generate Slideshow"
   - Application auto-detects dataset type
   - Creates annotated frames
   - Generates final video (2 seconds per sample minimum)
   - Video saved as `slideshow_video.mp4` in the dataset directory

### Example Datasets

#### Classification Dataset
```
dataset/
├── class_A/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── class_B/
    ├── image_001.jpg
    └── ...
```

#### Segmentation Dataset
```
dataset/
├── images/
│   ├── sample_001.jpg
│   ├── sample_002.jpg
│   └── ...
└── masks/
    ├── sample_001.png
    ├── sample_002.png
    └── ...
```

#### Object Detection (COCO Format)
```
dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── annotations/
    ├── image_001.json (COCO format)
    └── ...
```

#### Object Detection (YOLO Format)
```
dataset/
├── images/
│   ├── image_001.jpg
│   └── ...
└── labels/
    ├── image_001.txt (YOLO format)
    └── ...
```

## Architecture

### Main Components

**`pyqt.py`** - Main Application
- PyQt5 GUI for user input
- Worker thread for async processing
- Coordinates detection, annotation, and video generation

**`dataset_detector.py`** - Smart Dataset Detection
- Scans directory structure
- Detects dataset type from annotations
- Identifies annotation formats
- Returns `DatasetInfo` object with metadata

**`annotation_handlers.py`** - Annotation Visualization (Factory Pattern)
- `AnnotationHandler` - Abstract base class
- `ClassificationHandler` - Displays class labels
- `SegmentationHandler` - Overlays segmentation masks
- `DetectionHandler` - Draws bounding boxes from various formats
- `PoseHandler` - Visualizes keypoints and skeleton
- `UnknownHandler` - Fallback for unrecognized types
- `AnnotationHandlerFactory` - Creates appropriate handler

**`video_generator.py`** - Video Creation
- Creates MP4 video from frame sequences
- Grid layout support
- Configurable FPS and codec

### Factory Pattern Implementation

The codebase uses the **Factory Pattern** for extensible annotation handling:

```python
# In annotation_handlers.py
class AnnotationHandlerFactory:
    def __init__(self):
        self.handlers = {
            'classification': ClassificationHandler,
            'segmentation': SegmentationHandler,
            'detection': DetectionHandler,
            'pose': PoseHandler,
        }
    
    def register_handler(self, dataset_type: str, handler_class):
        """Register custom handler"""
        self.handlers[dataset_type] = handler_class
    
    def get_handler(self, dataset_type: str, dataset_info):
        """Get appropriate handler"""
        handler_class = self.handlers.get(dataset_type, UnknownHandler)
        return handler_class(dataset_info)
```

### Adding Custom Handlers

Extend the system for custom dataset types:

```python
# Custom handler
from annotation_handlers import AnnotationHandler

class CustomDatasetHandler(AnnotationHandler):
    def annotate_image(self, image_path: str, grid_cells: int) -> np.ndarray:
        img = self._load_image(image_path)
        # Your custom annotation logic
        return img

# Register in application
factory = AnnotationHandlerFactory()
factory.register_handler('custom_type', CustomDatasetHandler)
```

## Supported Annotation Formats

### Classification
- Directory-based (classes as folder names)
- JSON metadata files

### Segmentation
- PNG mask files (co-located or separate)
- Encoded as grayscale or colored images

### Object Detection
**COCO Format** (JSON)
```json
{
  "images": [...],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h], ...}
  ]
}
```

**YOLO Format** (TXT)
```
class_id center_x center_y width height
```

**Pascal VOC** (XML)
```xml
<annotation>
  <object>
    <name>class_name</name>
    <bndbox>
      <xmin>x1</xmin>
      <ymin>y1</ymin>
      <xmax>x2</xmax>
      <ymax>y2</ymax>
    </bndbox>
  </object>
</annotation>
```

**LabelMe Format** (JSON)
```json
{
  "shapes": [
    {"shape_type": "rectangle", "label": "object", "points": [[x1, y1], [x2, y2]]}
  ]
}
```

### Pose Estimation
- COCO keypoints format (JSON)
- Custom keypoint arrays

## Output

After processing, you'll find:
- **`slideshow_video.mp4`** - Generated video with all samples
  - 2+ seconds per sample
  - Grid layout (customizable size)
  - Annotated with relevant information

## Advanced Usage

### Programmatic API

```python
from dataset_detector import DatasetDetector
from annotation_handlers import AnnotationHandlerFactory
from video_generator import VideoGenerator

# Detect dataset
detector = DatasetDetector()
dataset_info = detector.detect_dataset('/path/to/dataset')

# Create handler
factory = AnnotationHandlerFactory()
handler = factory.get_handler(dataset_info.dataset_type, dataset_info)

# Process images
annotated_images = []
for image_path in dataset_info.image_paths[:10]:
    annotated = handler.annotate_image(image_path, grid_size=3)
    annotated_images.append(annotated)

# Generate video
video_gen = VideoGenerator(fps=30)
video_gen.create_video(annotated_images, 'output.mp4')
```

## Configuration

### Video Settings
Edit `VideoGenerator` in `video_generator.py`:
```python
video_gen = VideoGenerator(fps=30, codec='mp4v')  # Adjust FPS or codec
```

### Detection Parameters
Adjust in `dataset_detector.py`:
- `max_depth`: Directory scan depth
- `IMAGE_EXTENSIONS`: Supported image formats
- Match ratio thresholds for format detection

## Troubleshooting

**No images found**
- Ensure images are in supported formats (.jpg, .png, .bmp, .gif, .webp)
- Check directory permissions

**Dataset type detected as "Unknown"**
- Application couldn't match annotation format
- Verify annotation files are in supported format
- Add custom handler for your format

**Video generation fails**
- Check available disk space
- Ensure write permissions to dataset directory
- Verify OpenCV can access frame data

**Missing annotations**
- Verify annotation files match image names
- Check they're in supported formats
- Images and annotations can be in same or separate directories

## Performance Notes

- Large datasets: Processing scales with grid size
- Video generation: Real-time as frames are processed
- Memory: Holds current batch of frames in memory
- Threading: UI remains responsive during processing

## Dependencies

- **PyQt5**: GUI framework
- **OpenCV (cv2)**: Image processing
- **NumPy**: Array operations
- **Pillow**: Image I/O

## Future Enhancements

- [ ] Multi-threaded frame processing
- [ ] Support for video input datasets
- [ ] 3D pose visualization
- [ ] Instance segmentation support
- [ ] Custom color schemes
- [ ] Batch processing mode

## License

MIT

## Contact

For issues, suggestions, or contributions, please open an issue.
