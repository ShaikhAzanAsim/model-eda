"""
Annotation Handlers with Factory Pattern
Handles visualization of different annotation types:
- Classification
- Segmentation
- Object Detection
- Pose Estimation
"""

import os
import json
import random
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class AnnotationHandler(ABC):
    """Abstract base class for annotation handlers"""
    
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.coco_image_cache = {}
        
        # Build global COCO cache if format is coco
        if getattr(dataset_info, 'annotation_format', '') == 'coco':
            try:
                for ann_file in getattr(dataset_info, 'annotation_paths', []):
                    if ann_file.endswith('.json'):
                        with open(ann_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        img_id_to_name = {img['id']: img['file_name'] for img in data.get('images', [])}
                        for a in data.get('annotations', []):
                            fname = img_id_to_name.get(a['image_id'])
                            if fname:
                                if fname not in self.coco_image_cache:
                                    self.coco_image_cache[fname] = []
                                self.coco_image_cache[fname].append(a)
            except Exception:
                pass
        
    @abstractmethod
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60) -> np.ndarray:
        """
        Annotate image with dataset-specific information
        Returns annotated image as numpy array
        """
        pass
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image as BGR numpy array"""
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL for formats OpenCV might not support
            img = Image.open(image_path).convert('RGB')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
        
    def _extract_random_video_frames(self, video_path: str, num_frames: int = 60) -> list:
        """Extract a continuous sequence of random frames from a video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []
            
        start_frame = random.randint(0, max(0, total_frames - num_frames))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def _find_annotation_for_image(self, image_path: str, extension: str = None):
        """Find matching annotation file for image.
        Searches: sibling labels dir, all known annotation dirs, same dir.
        """
        image_stem = Path(image_path).stem
        image_dir = os.path.dirname(image_path)
        
        # Build list of candidate directories to search
        search_dirs = []
        
        # 1) Sibling directory: replace 'images' with 'labels' in path
        #    e.g. .../train/images/ -> .../train/labels/
        if 'images' in image_dir:
            sibling_labels = image_dir.replace('images', 'labels')
            if os.path.isdir(sibling_labels):
                search_dirs.append(sibling_labels)
        
        # 2) All known annotation directories from dataset detection
        if hasattr(self.dataset_info, 'annotation_dirs'):
            for d in self.dataset_info.annotation_dirs:
                if d not in search_dirs and os.path.isdir(d):
                    search_dirs.append(d)
        
        # 3) The primary annotation directory
        annotation_dir = self.dataset_info.annotation_dir
        if annotation_dir not in search_dirs and os.path.isdir(annotation_dir):
            search_dirs.append(annotation_dir)
        
        # 4) Same directory as image
        if image_dir not in search_dirs and os.path.isdir(image_dir):
            search_dirs.append(image_dir)
        
        # Search each directory for a matching annotation file
        for search_dir in search_dirs:
            # Try exact stem match first (faster)
            if extension:
                candidate = os.path.join(search_dir, image_stem + extension)
                if os.path.isfile(candidate):
                    return candidate
            
            # Fallback: scan directory for prefix match
            try:
                for file in os.listdir(search_dir):
                    if file.startswith(image_stem):
                        full_path = os.path.join(search_dir, file)
                        if extension is None or file.endswith(extension):
                            return full_path
            except PermissionError:
                continue
        
        return None
    
    def _add_label_text(self, img: np.ndarray, text: str) -> np.ndarray:
        """Add text label to image"""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw background rectangle
        x, y = 10, 30
        cv2.rectangle(img, (x - 5, y - text_size[1] - 5),
                     (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
        
        # Put text
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
        
        return img


class ClassificationHandler(AnnotationHandler):
    """Handler for classification datasets"""
    
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60):
        class_label = self._extract_class_label(image_path)
        ext = os.path.splitext(image_path)[1].lower()

        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            frames = self._extract_random_video_frames(image_path, num_frames=num_frames)
            annotated_frames = []
            for img in frames:
                if class_label:
                    img = self._add_label_text(img, f"Class: {class_label}")
                img = cv2.resize(img, (512, 512))
                annotated_frames.append(img)
            return annotated_frames
        else:
            img = self._load_image(image_path).copy()
            if class_label:
                img = self._add_label_text(img, f"Class: {class_label}")
            img = cv2.resize(img, (512, 512))
            return img
    
    def _extract_class_label(self, image_path: str) -> str:
        """Extract class label from directory name or metadata"""
        # Try to get from directory name
        parent_dir = os.path.basename(os.path.dirname(image_path))
        
        if parent_dir and parent_dir.isdigit() and hasattr(self.dataset_info, 'class_names'):
            class_idx = int(parent_dir)
            if class_idx in self.dataset_info.class_names:
                return str(self.dataset_info.class_names[class_idx])
            return parent_dir
        elif parent_dir and not parent_dir.isdigit():
            return parent_dir
        
        # Try to find in JSON metadata
        json_path = self._find_annotation_for_image(image_path, '.json')
        if json_path:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'class' in data:
                        return str(data['class'])
                    elif 'label' in data:
                        return str(data['label'])
            except:
                pass
        
        return parent_dir if parent_dir else "Unknown"


class SegmentationHandler(AnnotationHandler):
    """Handler for segmentation datasets"""
    
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60) -> np.ndarray:
        img = self._load_image(image_path).copy()
        
        # Find mask image
        mask_path = self._find_annotation_for_image(image_path)
        
        if mask_path and os.path.exists(mask_path):
            mask = self._load_image(mask_path)
            
            # Create overlay
            h, w = img.shape[:2]
            mask = cv2.resize(mask, (w, h))
            
            # Convert mask to single channel if needed
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Create colored overlay
            mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            
            # Blend
            img = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
            img = self._add_label_text(img, "Segmentation Mask")
        else:
            img = self._add_label_text(img, "Segmentation")
        
        img = cv2.resize(img, (512, 512))
        return img


class DetectionHandler(AnnotationHandler):
    """Handler for object detection datasets (COCO, YOLO, PascalVOC, LabelMe)"""
    
    # Distinct colors for different classes (BGR)
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (128, 0, 255), (0, 128, 255), (255, 0, 128), (0, 255, 128),
    ]
    
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60) -> np.ndarray:
        img = self._load_image(image_path).copy()
        h, w = img.shape[:2]
        
        # Get class name mapping from dataset info
        class_names = getattr(self.dataset_info, 'class_names', {})
        
        # Try different annotation formats (pass h, w to avoid reloading)
        annotations = self._get_annotations(image_path, w, h)
        
        if annotations:
            labels = []
            for ann in annotations:
                self._draw_bbox(img, ann, h, w, class_names)
                
                raw_class = ann.get('class', 'Unknown')
                if class_names and isinstance(raw_class, int) and raw_class in class_names:
                    labels.append(str(class_names[raw_class]))
                else:
                    labels.append(str(raw_class))
            
            num_objects = len(annotations)
            
            from collections import Counter
            counts = Counter(labels)
            label_parts = [f"{lbl}x{cnt}" if cnt > 1 else lbl for lbl, cnt in counts.items()]
            label_str = ", ".join(label_parts)
            
            if len(label_str) > 60:
                label_str = label_str[:57] + "..."
                
            img = self._add_label_text(img, f"Objects: {num_objects} ({label_str})")
        else:
            img = self._add_label_text(img, "Detection (no annotations)")
        
        img = cv2.resize(img, (512, 512))
        return img
    
    def _get_annotations(self, image_path: str, img_w: int, img_h: int) -> list:
        """Extract annotations from various formats"""
        
        # Try YOLO format first (most common for Roboflow datasets)
        yolo_path = self._find_annotation_for_image(image_path, '.txt')
        if yolo_path:
            annotations = self._parse_yolo_annotations(yolo_path, img_w, img_h)
            if annotations:
                return annotations
                
        # Check global COCO cache if applicable
        if getattr(self.dataset_info, 'annotation_format', '') == 'coco':
            fname = os.path.basename(image_path)
            if fname in self.coco_image_cache:
                return self._parse_coco_annotations({'annotations': self.coco_image_cache[fname]})
        
        # Try LabelMe format or local COCO (also .json)
        json_path = self._find_annotation_for_image(image_path, '.json')
        if json_path:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'annotations' in data:
                        return self._parse_coco_annotations(data)
                    if 'shapes' in data:
                        return self._parse_labelme_annotations(data['shapes'])
            except Exception:
                pass
        
        return []
    
    @staticmethod
    def _parse_coco_annotations(data: dict) -> list:
        """Parse COCO format annotations"""
        annotations = []
        for ann in data.get('annotations', []):
            extracted = {'class': ann.get('category_id', 'Unknown')}
            if 'bbox' in ann:
                x, y, width, height = ann['bbox']
                extracted['type'] = 'bbox'
                extracted['x'] = int(x)
                extracted['y'] = int(y)
                extracted['w'] = int(width)
                extracted['h'] = int(height)
            
            if 'segmentation' in ann and isinstance(ann['segmentation'], list) and ann['segmentation']:
                extracted['segmentation'] = ann['segmentation']
                # Mark as segmentation if bbox wasn't present
                if 'type' not in extracted:
                    extracted['type'] = 'segmentation'
                    
            if 'type' in extracted:
                annotations.append(extracted)
        return annotations
    
    @staticmethod
    def _parse_yolo_annotations(yolo_path: str, img_w: int, img_h: int) -> list:
        """Parse YOLO format annotations"""
        annotations = []
        try:
            with open(yolo_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx = float(parts[1]) * img_w
                        cy = float(parts[2]) * img_h
                        w = float(parts[3]) * img_w
                        h = float(parts[4]) * img_h
                        
                        annotations.append({
                            'type': 'bbox',
                            'x': int(cx - w/2),
                            'y': int(cy - h/2),
                            'w': int(w),
                            'h': int(h),
                            'class': class_id
                        })
        except Exception:
            pass
        
        return annotations
    
    @staticmethod
    def _parse_labelme_annotations(shapes: list) -> list:
        """Parse LabelMe format annotations"""
        annotations = []
        for shape in shapes:
            if shape.get('shape_type') == 'rectangle':
                points = shape.get('points', [])
                if len(points) >= 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    annotations.append({
                        'type': 'bbox',
                        'x': int(min(x1, x2)),
                        'y': int(min(y1, y2)),
                        'w': int(abs(x2 - x1)),
                        'h': int(abs(y2 - y1)),
                        'class': shape.get('label', 'Object')
                    })
        return annotations
    
    def _draw_bbox(self, img: np.ndarray, annotation: dict, img_h: int, img_w: int,
                   class_names: dict = None):
        """Draw bounding box and optional polygons on image with class-specific color and label"""
        if annotation.get('type') in ['bbox', 'segmentation']:
            # Pick color based on class id
            raw_class = annotation.get('class', 'Unknown')
            if isinstance(raw_class, int):
                color = self.COLORS[raw_class % len(self.COLORS)]
            else:
                color = self.COLORS[hash(str(raw_class)) % len(self.COLORS)]
                
            # Draw Segmentation Polygons
            if 'segmentation' in annotation:
                overlay = img.copy()
                for poly in annotation['segmentation']:
                    if isinstance(poly, list) and len(poly) >= 6:
                        # Reshape to (N, 1, 2)
                        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            
            if 'x' in annotation:
                x = max(0, annotation['x'])
                y = max(0, annotation['y'])
                w = annotation['w']
                h = annotation['h']
                
                x2 = min(img_w, x + w)
                y2 = min(img_h, y + h)
                
                thickness = 2
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness)
            else:
                # Fallback coordinates if no bbox
                if 'segmentation' in annotation and len(annotation['segmentation']) > 0:
                    poly = annotation['segmentation'][0]
                    if len(poly) >= 2:
                        x, y = int(poly[0]), int(poly[1])
                    else:
                        x, y = 10, 10
                else:
                    x, y = 10, 10
            
            # Resolve class name
            if class_names and isinstance(raw_class, int) and raw_class in class_names:
                label = class_names[raw_class]
            else:
                label = str(raw_class)
            
            # Draw label background + text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
            label_y = max(th + 4, y - 4)
            cv2.rectangle(img, (x, label_y - th - 4), (x + tw + 4, label_y + 2), color, -1)
            cv2.putText(img, label, (x + 2, label_y - 2),
                       font, font_scale, (255, 255, 255), text_thickness)


class PoseHandler(AnnotationHandler):
    """Handler for pose estimation datasets"""
    
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60) -> np.ndarray:
        img = self._load_image(image_path).copy()
        h, w = img.shape[:2]
        
        # Try to find keypoints
        keypoints = self._get_keypoints(image_path)
        
        if keypoints:
            self._draw_keypoints(img, keypoints, h, w)
            num_people = len(keypoints)
            img = self._add_label_text(img, f"People: {num_people}")
        else:
            img = self._add_label_text(img, "Pose")
        
        img = cv2.resize(img, (512, 512))
        return img
    
    def _get_keypoints(self, image_path: str) -> list:
        """Extract keypoints from annotation"""
        json_path = self._find_annotation_for_image(image_path, '.json')
        
        if not json_path:
            return []
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # COCO format
            if 'annotations' in data:
                keypoints_list = []
                for ann in data.get('annotations', []):
                    if 'keypoints' in ann:
                        keypoints_list.append(ann['keypoints'])
                return keypoints_list
            
            # Custom format
            if 'keypoints' in data:
                return [data['keypoints']]
        
        except:
            pass
        
        return []
    
    @staticmethod
    def _draw_keypoints(img: np.ndarray, keypoints_list: list, h: int, w: int):
        """Draw keypoints on image"""
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        for person_idx, keypoints in enumerate(keypoints_list):
            color = colors[person_idx % len(colors)]
            
            # Reshape keypoints if in flat format
            if isinstance(keypoints, list) and len(keypoints) > 0:
                if isinstance(keypoints[0], (int, float)):
                    # Flat format: [x1, y1, conf1, x2, y2, conf2, ...]
                    points = []
                    for i in range(0, len(keypoints), 3):
                        if i + 1 < len(keypoints):
                            x, y = int(keypoints[i]), int(keypoints[i + 1])
                            conf = keypoints[i + 2] if i + 2 < len(keypoints) else 1.0
                            if conf > 0.5:  # Only draw confident points
                                points.append((x, y))
                else:
                    points = [(int(p[0]), int(p[1])) for p in keypoints if len(p) >= 2]
                
                # Draw connections (skeleton)
                for i in range(len(points) - 1):
                    cv2.line(img, points[i], points[i + 1], color, 2)
                
                # Draw keypoints
                for point in points:
                    cv2.circle(img, point, 5, color, -1)
                    cv2.circle(img, point, 5, (255, 255, 255), 1)


class UnknownHandler(AnnotationHandler):
    """Fallback handler for unknown dataset types"""
    
    def annotate_image(self, image_path: str, grid_cells: int,
                       num_frames: int = 60) -> np.ndarray:
        img = self._load_image(image_path).copy()
        img = self._add_label_text(img, "Unknown Dataset Type")
        img = cv2.resize(img, (512, 512))
        return img


class AnnotationHandlerFactory:
    """Factory for creating appropriate annotation handlers"""
    
    def __init__(self):
        self.handlers = {
            'classification': ClassificationHandler,
            'segmentation': SegmentationHandler,
            'detection': DetectionHandler,
            'pose': PoseHandler,
            'unknown': UnknownHandler,
        }
    
    def register_handler(self, dataset_type: str, handler_class):
        """Register a custom handler for a new dataset type"""
        self.handlers[dataset_type] = handler_class
    
    def get_handler(self, dataset_type: str, dataset_info) -> AnnotationHandler:
        """Get appropriate handler for dataset type"""
        handler_class = self.handlers.get(dataset_type.lower(), UnknownHandler)
        return handler_class(dataset_info)
