# Project Structure Overview

## Directory Layout

```
eda_pipeline/
│
├── Pipeline/                          # Main codebase package
│   ├── __init__.py                    # Package initialization
│   ├── pyqt.py                        # Main PyQt5 application (GUI)
│   ├── dataset_detector.py            # Dataset detection & analysis
│   ├── annotation_handlers.py         # Annotation handler (Factory pattern)
│   └── video_generator.py             # MP4 video creation
│
├── Documentation/
│   ├── README_VISUALIZER.md           # Complete feature documentation
│   ├── QUICK_START.md                 # 5-minute setup guide
│   ├── ARCHITECTURE.md                # Design patterns & architecture
│   ├── CONFIGURATION.md               # Tunable parameters
│   └── PROJECT_STRUCTURE.md           # This file
│
├── Scripts/
│   ├── examples.py                    # 5 usage examples
│   └── verify_installation.py         # Installation verification
│
├── requirements.txt                   # Python dependencies
│
└── Data Folders/
    ├── imagenet/                      # Example dataset (your existing)
    ├── minicoco/                      # Example dataset (your existing)
    └── segment/                       # Example dataset (your existing)
```

## File Descriptions

### Core Application Files (Pipeline/)

#### `pyqt.py` - Main Application (420 lines)
- **Purpose**: PyQt5 GUI and orchestration
- **Classes**:
  - `VisualizationWorker(QThread)`: Background processing
  - `DatasetVisualizer(QWidget)`: Main window
- **Key Features**:
  - Parameter input (directory, N, grid size)
  - Non-blocking worker thread
  - Progress monitoring
  - Error handling

#### `dataset_detector.py` - Dataset Analysis (340 lines)
- **Purpose**: Auto-detect dataset type and structure
- **Classes**:
  - `DatasetInfo`: Data container
  - `DatasetDetector`: Main detection engine
- **Capabilities**:
  - Recursive directory scanning
  - Image/annotation matching
  - Format identification (COCO, YOLO, Pascal VOC, LabelMe)
  - Dataset type classification

#### `annotation_handlers.py` - Annotation Rendering (420 lines)
- **Purpose**: Visualize annotations (Factory pattern)
- **Classes**:
  - `AnnotationHandler`: Abstract base
  - `ClassificationHandler`: Label display
  - `SegmentationHandler`: Mask overlay
  - `DetectionHandler`: Bounding boxes
  - `PoseHandler`: Keypoint visualization
  - `UnknownHandler`: Fallback
  - `AnnotationHandlerFactory`: Factory class
- **Pattern**: Factory for extensibility
- **Key Features**:
  - Multiple annotation format support
  - Easy custom handler registration
  - Consistent interface

#### `video_generator.py` - Video Creation (160 lines)
- **Purpose**: Convert frames to MP4 video
- **Classes**:
  - `VideoGenerator`: Video creation engine
- **Methods**:
  - `create_video()`: Main video generation
  - `create_grid_video()`: Grid layout videos
  - `_create_grid_frame()`: Image grid composition
  - `create_montage()`: Static image montage

#### `__init__.py` - Package Init (40 lines)
- **Purpose**: Package initialization
- **Exports**: Main classes for external use

### Documentation Files

#### `README_VISUALIZER.md` (350 lines)
Complete reference including:
- Features and capabilities
- Installation instructions
- Usage examples for each dataset type
- Annotation format specifications
- API reference
- Troubleshooting guide
- Performance notes

#### `QUICK_START.md` (200 lines)
Fast-track guide including:
- 5-minute installation
- Basic workflow
- Common use cases
- Example dataset structures
- Quick troubleshooting

#### `ARCHITECTURE.md` (500 lines)
Deep technical documentation:
- System overview diagrams
- Component descriptions
- Factory pattern explanation
- Data flow diagrams
- Configuration points
- Extension examples
- Design principles

#### `CONFIGURATION.md` (350 lines)
Parameter reference including:
- All tunable parameters
- Default values
- Performance settings
- Custom handler template
- Recommended configurations
- Benchmarks

### Example & Utility Files

#### `examples.py` (370 lines)
5 complete working examples:
1. Dataset detection and analysis
2. Annotating random samples
3. Complete slideshow generation
4. Custom handler implementation
5. Batch processing multiple datasets

#### `verify_installation.py` (280 lines)
Installation verification:
- Dependency checking
- Local module validation
- PyQt5 functionality tests
- OpenCV capability tests
- File structure verification
- Permission testing
- Detailed error reporting

### Configuration Files

#### `requirements.txt` (4 lines)
Python dependencies:
- PyQt5==5.15.7
- opencv-python==4.8.1.78
- numpy==1.24.3
- Pillow==10.0.0

---

## File Statistics

```
Component           Files  Lines  Purpose
─────────────────────────────────────────────
Core Application      5     1020  Main system
Documentation         4     1400  User guidance
Examples              1      370  Demonstrations
Utilities             2      320  Verification
Config               1        4  Dependencies
─────────────────────────────────────────────
Total               13     3114  Complete system
```

## Data Flow Between Files

```
User Input (pyqt.py)
        │
        ▼
DatasetDetector (dataset_detector.py)
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
DatasetInfo                        AnnotationHandlerFactory
(dataset_detector.py)             (annotation_handlers.py)
        │                                 │
        └─────────────────────────────────┘
                      │
                      ▼
             Specific Handler
          (ClassificationHandler,
           SegmentationHandler, etc.)
                      │
                      ▼
            Annotated Images (numpy arrays)
                      │
                      ▼
            VideoGenerator (video_generator.py)
                      │
                      ▼
            Output MP4 File (slideshow_video.mp4)
```

## Dependencies Graph

```
pyqt.py
  ├── PyQt5
  ├── dataset_detector.py
  │   ├── cv2 (OpenCV)
  │   ├── numpy
  │   └── json, os, pathlib
  ├── annotation_handlers.py
  │   ├── cv2 (OpenCV)
  │   ├── numpy
  │   ├── PIL (Pillow)
  │   └── json, abc
  └── video_generator.py
      ├── cv2 (OpenCV)
      └── numpy
```

## How to Extend

### Add New Dataset Type Support

1. **Create Handler** → `annotation_handlers.py`
   ```python
   class MyHandler(AnnotationHandler):
       def annotate_image(self, image_path, grid_cells):
           # Your logic
   ```

2. **Register Handler** → Use factory
   ```python
   factory.register_handler('my_type', MyHandler)
   ```

3. **Update Detection** → `dataset_detector.py`
   - Add format detection logic to `_detect_dataset_type()`
   - Add annotation parsing to `_infer_from_annotation_content()`

### Add New Annotation Format

1. **Extend Detection** → `dataset_detector.py`
   - Add extension to `ANNOTATION_EXTENSIONS`
   - Add parsing logic

2. **Update Handlers** → `annotation_handlers.py`
   - Add parsing method in relevant handler
   - Call parser when annotation found

### Customize Video Output

1. **Modify Generator** → `video_generator.py`
   - Adjust `fps`, `codec`
   - Customize `_create_grid_frame()`

2. **Configure in App** → `pyqt.py`
   - Pass custom parameters to `VideoGenerator`

---

## Testing Checklist

Before deploying to production:

- [ ] Run `verify_installation.py` successfully
- [ ] Test with classification dataset
- [ ] Test with segmentation dataset
- [ ] Test with detection dataset (COCO format)
- [ ] Test with detection dataset (YOLO format)
- [ ] Test with pose estimation dataset
- [ ] Test with custom directory structure
- [ ] Test with co-located images and annotations
- [ ] Test with separate directories
- [ ] Test various grid sizes (2x2 to 6x6)
- [ ] Test various sample sizes (1, 10, 100)
- [ ] Verify video output quality
- [ ] Check error handling
- [ ] Test on different OS (if applicable)

---

## Development Notes

### Code Style
- PEP 8 compliant
- Type hints throughout
- Comprehensive docstrings
- Clear variable names

### Error Handling
- Graceful degradation
- User-friendly error messages
- Logging for debugging
- Exception handling in critical sections

### Performance
- Non-blocking UI (threading)
- Efficient image processing
- Memory-conscious design
- Scalable architecture

### Extensibility
- Factory pattern for handlers
- Abstract base classes
- Plugin-friendly architecture
- Easy to add new formats

---

## File Relationships

```
┌──────────────────────────────────────┐
│      USER (pyqt.py)                 │
│   - Input: directory, N, GRID_SIZE  │
│   - Output: Video generated         │
└──────────┬───────────────────────────┘
           │
           ├─ Detects ─────────► dataset_detector.py
           │                      (dataclass)
           │                      DatasetInfo
           │
           ├─ Creates Handler ──► annotation_handlers.py
           │                      (Factory Pattern)
           │                      - Classification
           │                      - Segmentation
           │                      - Detection
           │                      - Pose
           │
           └─ Builds Video ─────► video_generator.py
                                   (MP4 Creation)

examples.py ─► Shows usage of above components
verify_installation.py ─► Tests setup
```

---

## Quick Reference

| Task | File | Function |
|------|------|----------|
| Run GUI | `Pipeline/pyqt.py` | Run directly |
| Detect dataset | `dataset_detector.py` | `DatasetDetector.detect_dataset()` |
| Get handler | `annotation_handlers.py` | `AnnotationHandlerFactory.get_handler()` |
| Annotate image | Handler classes | `annotate_image()` method |
| Create video | `video_generator.py` | `VideoGenerator.create_video()` |
| Learn usage | `examples.py` | 5 complete examples |
| Verify setup | `verify_installation.py` | Run and check output |
| Read docs | `README_VISUALIZER.md` | Complete reference |
| Get started | `QUICK_START.md` | 5-minute guide |
| Understand design | `ARCHITECTURE.md` | Design patterns |
| Configure | `CONFIGURATION.md` | Parameters |

---

## Size & Complexity Metrics

```
File                          Lines  Complexity  Purpose
───────────────────────────────────────────────────────────
pyqt.py                        280    Medium     GUI/Orchestration
dataset_detector.py            340    High       Complex logic
annotation_handlers.py         420    High       Multiple handlers
video_generator.py             160    Low        Video creation
───────────────────────────────────────────────────────────
Core Total                    1200    Medium     Main system

examples.py                    370    Medium     Examples
verify_installation.py         280    Low        Verification
───────────────────────────────────────────────────────────
Total Code                    1850    Medium     Complete
───────────────────────────────────────────────────────────
Documentation                 1400    Low        Guides
───────────────────────────────────────────────────────────
Full Project                  3250    Medium     Everything
```

---

## Maintenance

### Regular Updates
- Update dependencies in `requirements.txt`
- Add new annotation formats as they emerge
- Improve detection logic based on real datasets
- Optimize performance for large datasets

### Common Modifications
- Add custom handlers (extend `AnnotationHandler`)
- Adjust parameters (see `CONFIGURATION.md`)
- Change default sizes (in each handler)
- Modify video settings (`VideoGenerator`)

### Troubleshooting Guide
- Check `QUICK_START.md` for common issues
- Run `verify_installation.py`
- Review `ARCHITECTURE.md` for understanding flow
- Check error messages in status/log

