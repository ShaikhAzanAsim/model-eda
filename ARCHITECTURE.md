# Architecture & Design Document

## System Overview

The Image Dataset Visualizer is a modular, extensible system for analyzing image datasets using design patterns and factory-based architecture.

```
┌─────────────────────────────────────────────────────────────┐
│                         PyQt5 GUI                           │
│  (Directory Selection, Parameters, Progress Monitoring)     │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
┌──────────────────┐  │  ┌──────────────────────┐
│  DatasetDetector │  │  │ AnnotationHandler    │
│                  │  │  │ Factory Pattern      │
│ - Scans files    │  │  │                      │
│ - Identifies     │  │  │ Handlers:            │
│   format         │  │  │ - Classification     │
│ - Detects type   │  │  │ - Segmentation      │
└──────────────────┘  │  │ - Detection         │
                      │  │ - Pose              │
                      │  │ - Custom            │
                      │  └──────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  VideoGenerator      │
            │                      │
            │ - Compose frames     │
            │ - Write MP4          │
            │ - 30fps, configurable│
            └──────────────────────┘
                      │
                      ▼
            ┌──────────────────────┐
            │  Output Video        │
            │  (slideshow.mp4)     │
            └──────────────────────┘
```

## Core Components

### 1. PyQt Application (`pyqt.py`)

**Purpose**: User interface and orchestration

**Classes**:
- `VisualizationWorker(QThread)`: Background processing thread
  - Calls detector, handler factory, and video generator
  - Emits progress signals to UI
  - Handles exceptions gracefully

- `DatasetVisualizer(QWidget)`: Main application window
  - Parameter input (directory, N, grid size)
  - Progress monitoring
  - Result display/error handling

**Key Features**:
- Non-blocking UI (worker thread)
- Real-time progress updates
- Error handling and user feedback

---

### 2. Dataset Detection (`dataset_detector.py`)

**Purpose**: Automatically identify dataset type and structure

**Classes**:
- `DatasetInfo`: Data class containing detected information
  - `dataset_type`: Classification/Segmentation/Detection/Pose/Unknown
  - `image_paths`: List of all image file paths
  - `annotation_paths`: Dict of annotation lists by format
  - `annotation_format`: Detected format (COCO/YOLO/Pascal VOC/LabelMe)
  - `has_separate_dirs`: Boolean for directory structure
  - `metadata`: Additional information

- `DatasetDetector`: Main detection class
  - **Methods**:
    - `detect_dataset()`: Main entry point
    - `_scan_directory()`: Recursive file discovery
    - `_determine_directory_structure()`: Identify image/annotation layout
    - `_detect_dataset_type()`: Classify dataset
    - `_infer_from_annotation_content()`: Parse annotation files

**Detection Logic**:
1. Recursively scan directory tree (max 3 levels deep)
2. Identify image vs annotation files
3. Check for specific annotation patterns:
   - Segmentation: Image-mask pairs (PNG files)
   - Detection: COCO (JSON), YOLO (TXT), Pascal VOC (XML), LabelMe (JSON)
   - Pose: COCO with keypoints
   - Classification: Directory names or JSON metadata
4. Prioritize format detection (segmentation > pascal_voc > coco > yolo > labelme > none)
5. Return `DatasetInfo` with full metadata

**Supported Formats**:
```
COCO JSON:
{
  "images": [...],
  "annotations": [{"id": 1, "bbox": [x,y,w,h], "keypoints": [...]}]
}

YOLO TXT:
class_id center_x center_y width height

Pascal VOC XML:
<annotation>
  <object>
    <bndbox>
      <xmin>x1</xmin>
      ...
    </bndbox>
  </object>
</annotation>

LabelMe JSON:
{
  "shapes": [
    {"shape_type": "rectangle", "points": [[x1,y1],[x2,y2]]}
  ]
}

Segmentation Masks:
PNG files with matching base names
```

---

### 3. Annotation Handlers - Factory Pattern (`annotation_handlers.py`)

**Architecture Pattern**: **Factory Pattern**

The factory pattern allows runtime selection and registration of handlers without modifying core code.

```
                    AnnotationHandlerFactory
                           │
            ┌──────────────┬┼┬──────────────┐
            │              │ │              │
            ▼              ▼▼              ▼
      ClassificationHandler  SegmentationHandler
            │
            ├─ handler 1
            ├─ handler 2
            └─ ... n custom handlers

    factory.register_handler('custom_type', CustomHandler)
    handler = factory.get_handler(type, info)
```

**Base Class**:
```python
class AnnotationHandler(ABC):
    @abstractmethod
    def annotate_image(self, image_path, grid_cells) -> np.ndarray:
        """Returns annotated image"""
        pass
    
    def _load_image(self, path) -> np.ndarray:
        """Load image as BGR array"""
    
    def _find_annotation_for_image(self, image_path):
        """Find matching annotation file"""
    
    def _add_label_text(self, img, text) -> np.ndarray:
        """Render text on image"""
```

**Concrete Handlers**:

1. **ClassificationHandler**
   - Extracts class label from directory name or JSON
   - Adds "Class: {label}" text to image
   - Resizes to 512x512

2. **SegmentationHandler**
   - Finds mask PNG file with matching name
   - Overlays mask on image with colormap (70% image, 30% mask)
   - Resizes to 512x512

3. **DetectionHandler**
   - Supports multiple formats: COCO, YOLO, Pascal VOC, LabelMe
   - Draws bounding boxes with class labels
   - Shows object count ("Objects: 5")
   - Resizes to 512x512

4. **PoseHandler**
   - Extracts keypoints from JSON annotations
   - Draws skeleton (connections between joints)
   - Draws circles for individual keypoints
   - Color-coded by person
   - Resizes to 512x512

5. **UnknownHandler**
   - Fallback for unrecognized dataset types
   - Displays "Unknown Dataset Type" label

**Factory Implementation**:
```python
class AnnotationHandlerFactory:
    def __init__(self):
        self.handlers = {
            'classification': ClassificationHandler,
            'segmentation': SegmentationHandler,
            'detection': DetectionHandler,
            'pose': PoseHandler,
            'unknown': UnknownHandler,
        }
    
    def register_handler(self, dataset_type: str, handler_class):
        """Register new handler - enables custom types"""
        self.handlers[dataset_type] = handler_class
    
    def get_handler(self, dataset_type: str, dataset_info):
        """Get appropriate handler - runtime polymorphism"""
        handler_class = self.handlers.get(
            dataset_type.lower(),
            UnknownHandler
        )
        return handler_class(dataset_info)
```

**Benefits of Factory Pattern**:
1. **Extensibility**: Add new handlers without modifying core code
2. **Maintainability**: Each handler encapsulates its logic
3. **Decoupling**: Main code doesn't know about specific handlers
4. **Runtime Selection**: Choose handler based on detected type
5. **Easy Testing**: Mock handlers for unit tests

**Adding Custom Handler**:
```python
class MyCustomHandler(AnnotationHandler):
    def annotate_image(self, image_path, grid_cells):
        img = self._load_image(image_path)
        # Custom processing logic
        return img

# Register
factory = AnnotationHandlerFactory()
factory.register_handler('my_dataset_type', MyCustomHandler)
```

---

### 4. Video Generator (`video_generator.py`)

**Purpose**: Convert annotated frames into MP4 video

**Classes**:
- `VideoGenerator`: Creates video files
  - **Parameters**:
    - `fps` (default 30): Frames per second
    - `codec` (default 'mp4v'): Video codec
  
  - **Methods**:
    - `create_video(frames, output_path)`: Main video creation
      - Takes list of numpy arrays (BGR)
      - Writes sequential frames
      - Returns True/False for success
    
    - `create_grid_video(image_groups, grid_size, output_path)`: Grid layout
      - Creates grid frame for each group
      - Each group is arranged in grid_size x grid_size
    
    - `_create_grid_frame(images, grid_size)`: Grid arrangement
      - Arranges images in grid on canvas
      - Handles variable number of images
      - Same cell size for all images
    
    - `create_montage(images, grid_size, output_path)`: Static image
      - Creates single static montage image instead of video

**Video Parameters**:
- FPS: 30 (standard)
- Codec: mp4v (H.264 compatible)
- Duration per sample: 60 frames (2 seconds)
- Resolution: Variable (first frame determines size)

**Frame Processing**:
1. Validate frame dimensions
2. Ensure BGR color space
3. Resize if necessary
4. Write to video file
5. Verify file creation

---

## Data Flow Diagram

```
User Input
  │
  ├─ Directory: "/path/to/dataset"
  ├─ N Samples: 10
  └─ Grid Size: 3x3
  │
  ▼
DatasetDetector.detect_dataset(path)
  │
  ├─ Scan directory recursively
  ├─ Find all images
  ├─ Find all annotations
  ├─ Match image-annotation pairs
  ├─ Infer dataset type
  └─ Return DatasetInfo
  │
  ▼
Get Annotation Handler
  │
  └─ factory.get_handler(type, info)
      Returns appropriate handler instance
  │
  ▼
For Each Random Sample (1 to N):
  │
  ├─ Load image
  ├─ Load annotations
  ├─ Handler.annotate_image(img_path, grid_size)
  │   ├─ Find annotation file
  │   ├─ Parse annotation
  │   ├─ Draw overlays (boxes, masks, keypoints, labels)
  │   └─ Resize to 512x512
  │   └─ Return annotated image
  ├─ Duplicate frame 60 times (2 seconds @ 30fps)
  └─ Add to frame list
  │
  ▼
VideoGenerator.create_video(frames, output_path)
  │
  ├─ Initialize video writer
  ├─ For each frame:
  │   ├─ Ensure correct shape
  │   ├─ Ensure BGR color space
  │   └─ Write to file
  ├─ Close video writer
  └─ Return success status
  │
  ▼
Output File
  └─ slideshow_video.mp4
     (N samples × 2 seconds = N×2 seconds total)
```

---

## Configuration Points

### Dataset Detection
- **Image extensions**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.webp`
- **Scan depth**: Max 3 levels (configurable in `_scan_directory`)
- **Segmentation match ratio**: 0.3 (30% of images should have matching masks)
- **Annotation formats**: COCO, YOLO, Pascal VOC, LabelMe, custom

### Annotation Handlers
- **Image size after annotation**: 512×512 (configurable per handler)
- **Label text font**: HERSHEY_SIMPLEX, scale 0.6
- **Segmentation overlay alpha**: 70% image, 30% mask
- **Detection bbox color**: Green (0,255,0), thickness 2
- **Pose keypoint size**: Radius 5, confidence threshold 0.5

### Video Generation
- **FPS**: 30 (configurable)
- **Codec**: mp4v (H.264)
- **Display time per sample**: 2 seconds (60 frames)
- **Frame format**: BGR, numpy uint8

---

## Error Handling

**Graceful Degradation**:
1. Image loading failure → Skip image, continue processing
2. Annotation not found → Show image without annotation
3. Invalid annotation format → Use handler for detected format
4. Video write failure → Return error, suggest trouble-shooting

**User Feedback**:
- Progress bar updates
- Status messages throughout process
- Error dialogs with actionable messages
- Logs printed to console

---

## Performance Considerations

- **Memory**: Holds batch of frames in memory (scales with grid size)
- **CPU**: Image processing dominates, scales with N and grid size
- **Disk**: Video file ~5-10 MB per minute (varies with resolution)
- **Threading**: Worker thread prevents UI blocking

**Optimization Tips**:
1. Reduce grid size for faster processing
2. Use smaller sample size for testing
3. Store dataset on fast storage (SSD preferred)
4. Adequate RAM (recommend 4GB+)

---

## Extension Examples

### Example 1: Custom Annotation Format
```python
class CustomJSONHandler(AnnotationHandler):
    def annotate_image(self, image_path, grid_cells):
        img = self._load_image(image_path)
        
        # Custom logic for your format
        ann_path = self._find_annotation_for_image(image_path, '.json')
        if ann_path:
            with open(ann_path) as f:
                data = json.load(f)
            # Process custom data structure
            # Draw on image
        
        return cv2.resize(img, (512, 512))

factory.register_handler('custom_json', CustomJSONHandler)
```

### Example 2: Multi-Model Visualization
```python
class InstanceSegmentationHandler(AnnotationHandler):
    """Handle instance segmentation with colored masks"""
    
    def annotate_image(self, image_path, grid_cells):
        img = self._load_image(image_path)
        masks = self._load_instance_masks(image_path)
        
        # Generate random colors for each instance
        for instance_id, mask in enumerate(masks):
            color = self._random_color()
            img = cv2.addWeighted(
                img, 0.7,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3,
                0
            )
        
        return cv2.resize(img, (512, 512))

factory.register_handler('instance_segmentation', InstanceSegmentationHandler)
```

---

## Testing Strategy

1. **Unit Tests**: Test individual handlers with known data
2. **Integration Tests**: Test detector with sample datasets
3. **Functional Tests**: End-to-end with real datasets
4. **Regression Tests**: Ensure new handlers don't break existing ones

---

## Future Enhancements

1. **Multi-threading**: Parallel frame processing
2. **3D Visualization**: Support for 3D pose and point clouds
3. **Batch Mode**: Process multiple datasets automatically
4. **Custom UI**: Per-dataset type visualization options
5. **Compression**: Configurable video quality/bitrate
6. **Formats**: Support for additional annotation formats
7. **Analytics**: Dataset statistics and quality metrics

---

## Design Principles

1. **SOLID Principles**:
   - **S**ingle Responsibility: Each handler focuses on one type
   - **O**pen/Closed: Open for extension, closed for modification (factory)
   - **L**iskov Substitution: All handlers implement same interface
   - **I**nterface Segregation: Minimal abstract base class
   - **D**ependency Inversion: Depend on abstractions, not concrete handlers

2. **DRY (Don't Repeat Yourself)**:
   - Common functionality in base class
   - Shared utility methods
   - Reusable helper functions

3. **Clean Code**:
   - Clear naming conventions
   - Comprehensive docstrings
   - Type hints for clarity
   - Error handling throughout

---

## Glossary

- **Dataset Type**: Classification, Segmentation, Detection, Pose
- **Annotation Format**: How annotations are stored (COCO, YOLO, etc.)
- **Handler**: Converts annotations to visual overlays
- **Factory**: Creates objects without exposing creation logic
- **Grid Size**: N×N layout (2×2 = 4-cell grid, 3×3 = 9-cell grid)
- **FPS**: Frames Per Second (video playback speed)
- **Codec**: Algorithm for video compression

