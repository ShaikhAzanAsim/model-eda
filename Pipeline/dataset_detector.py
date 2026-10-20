"""
Dataset Type Detection and Structure Analysis
Identifies: Classification, Segmentation, Object Detection, Pose Estimation, etc.
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import cv2
import numpy as np


@dataclass
class DatasetInfo:
    """Container for detected dataset information"""
    dataset_type: str  # 'classification', 'segmentation', 'detection', 'pose', 'unknown'
    image_paths: List[str]
    annotation_paths: List[str]
    image_dir: str
    annotation_dir: str
    annotation_format: str  # 'coco', 'pascal_voc', 'yolo', 'labelme', 'custom', etc.
    has_separate_dirs: bool
    metadata: Dict = None
    annotation_dirs: List[str] = field(default_factory=list)  # ALL annotation directories
    class_names: Dict[int, str] = field(default_factory=dict)  # class_id -> class_name mapping


class DatasetDetector:
    """Auto-detect dataset type and structure"""
    
    # Image and Video extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    # Annotation formats
    ANNOTATION_EXTENSIONS = {
        '.json': ['labelme', 'coco', 'custom'],
        '.xml': ['pascal_voc', 'vott'],
        '.txt': ['yolo', 'darknet'],
        '.mat': ['matlab'],
        '.png': ['segmentation_mask'],  # For segmentation masks
        '.npy': ['numpy'],
    }
    
    def __init__(self):
        self.image_paths = []
        self.annotation_paths = defaultdict(list)
        self.subdirs = []
        
    def detect_dataset(self, root_dir: str) -> Optional[DatasetInfo]:
        """
        Main detection method
        Returns DatasetInfo if dataset is found, None otherwise
        """
        if not os.path.isdir(root_dir):
            return None
        
        # Reset state
        self.image_paths = []
        self.annotation_paths = defaultdict(list)
        
        # Scan directory structure
        self._scan_directory(root_dir)
        
        if not self.image_paths:
            return None
        
        # Determine directory structure
        image_dir, annotation_dir, has_separate, all_annotation_dirs = self._determine_directory_structure(root_dir)
        
        # Detect dataset type
        dataset_type, annotation_format = self._detect_dataset_type(root_dir)
        
        # Gather annotations
        annotations = self.annotation_paths.get(annotation_format, [])
        
        # Extract class names based on architecture rules
        class_names = self._extract_class_names(root_dir, annotation_format, annotations)
        
        return DatasetInfo(
            dataset_type=dataset_type,
            image_paths=self.image_paths,
            annotation_paths=annotations,
            image_dir=image_dir,
            annotation_dir=annotation_dir,
            annotation_format=annotation_format,
            has_separate_dirs=has_separate,
            metadata={
                'total_images': len(self.image_paths),
                'total_annotations': len(annotations),
            },
            annotation_dirs=list(all_annotation_dirs),
            class_names=class_names,
        )
    
    def _scan_directory(self, root_dir: str, depth: int = 0, max_depth: int = 3):
        """Recursively scan directory for images and annotations"""
        if depth > max_depth:
            return
        
        try:
            entries = os.listdir(root_dir)
        except PermissionError:
            return
        
        for entry in entries:
            full_path = os.path.join(root_dir, entry)
            
            if os.path.isdir(full_path):
                self.subdirs.append(full_path)
                self._scan_directory(full_path, depth + 1, max_depth)
            elif os.path.isfile(full_path):
                ext = Path(full_path).suffix.lower()
                
                if ext in self.IMAGE_EXTENSIONS:
                    self.image_paths.append(full_path)
                
                if ext in self.ANNOTATION_EXTENSIONS:
                    for fmt in self.ANNOTATION_EXTENSIONS[ext]:
                        self.annotation_paths[fmt].append(full_path)
    
    def _determine_directory_structure(self, root_dir: str) -> tuple:
        """
        Determine if images and annotations are in same or separate directories
        Returns: (image_dir, annotation_dir, has_separate_dirs, all_annotation_dirs)
        """
        if not self.image_paths:
            return root_dir, root_dir, False, set()
        
        # Find common image directory
        image_dirs = set(os.path.dirname(p) for p in self.image_paths)
        
        # Find ALL annotation directories
        annotation_dirs = set()
        for paths in self.annotation_paths.values():
            annotation_dirs.update(os.path.dirname(p) for p in paths)
        
        # Pick a representative annotation dir (but keep all of them)
        image_dir = max(image_dirs, key=lambda x: len(x)) if image_dirs else root_dir
        annotation_dir = max(annotation_dirs, key=lambda x: len(x)) if annotation_dirs else root_dir
        
        has_separate = image_dir != annotation_dir
        
        return image_dir, annotation_dir, has_separate, annotation_dirs
    
    def _detect_dataset_type(self, root_dir: str) -> tuple:
        """
        Detect dataset type based on annotations found
        Returns: (dataset_type, annotation_format)
        """
        # Priority order for detection
        priority = ['segmentation_mask', 'pascal_voc', 'coco', 'yolo', 'labelme', 'custom']
        
        # Check for segmentation masks (PNG files with specific patterns)
        if self._has_segmentation_masks(root_dir):
            return 'segmentation', 'segmentation_mask'
        
        # Check annotation formats in priority order
        for fmt in priority:
            if fmt in self.annotation_paths and self.annotation_paths[fmt]:
                dataset_type = self._determine_type_from_format(fmt, root_dir)
                if dataset_type != 'unknown':
                    return dataset_type, fmt
        
        # Default to classification if only images found
        return 'classification', 'none'
    
    def _has_segmentation_masks(self, root_dir: str) -> bool:
        """Check if dataset contains segmentation masks"""
        mask_candidates = self.annotation_paths.get('segmentation_mask', [])
        
        if not mask_candidates:
            return False
        
        # Check if PNG files have similar names to images
        image_names = set(Path(p).stem for p in self.image_paths)
        mask_names = set(Path(p).stem for p in mask_candidates)
        
        # If significant overlap, likely segmentation dataset
        overlap = len(image_names & mask_names)
        match_ratio = overlap / len(image_names) if image_names else 0
        
        return match_ratio > 0.3
    
    def _determine_type_from_format(self, fmt: str, root_dir: str) -> str:
        """Determine dataset type from annotation format"""
        if fmt == 'segmentation_mask':
            return 'segmentation'
        
        elif fmt in ['coco', 'pascal_voc', 'yolo', 'darknet', 'labelme']:
            # Parse to check for specific fields
            return self._infer_from_annotation_content(fmt)
        
        else:
            return 'unknown'
    
    def _infer_from_annotation_content(self, fmt: str) -> str:
        """Infer type by examining annotation file content"""
        sample_annotations = self.annotation_paths.get(fmt, [])[:5]
        
        if not sample_annotations:
            return 'unknown'
        
        for ann_path in sample_annotations:
            try:
                if fmt in ['coco', 'labelme', 'custom'] and ann_path.endswith('.json'):
                    with open(ann_path, 'r') as f:
                        content = json.load(f)
                    
                    # Check COCO format
                    if 'images' in content and 'annotations' in content:
                        return self._determine_coco_type(content)
                    
                    # Check LabelMe format
                    if 'shapes' in content:
                        return self._determine_labelme_type(content['shapes'])
                
                elif fmt in ['pascal_voc'] and ann_path.endswith('.xml'):
                    with open(ann_path, 'r') as f:
                        content = f.read()
                    if 'difficult' in content or 'bndbox' in content:
                        return 'detection'
                
                elif fmt in ['yolo', 'darknet'] and ann_path.endswith('.txt'):
                    with open(ann_path, 'r') as f:
                        lines = f.readlines()
                    if lines:  # YOLO format typically has values
                        return 'detection'
            
            except (json.JSONDecodeError, IOError):
                continue
        
        return 'unknown'
    
    @staticmethod
    def _determine_coco_type(content: dict) -> str:
        """Determine type from COCO format"""
        if 'annotations' not in content or not content['annotations']:
            return 'classification'
        
        ann = content['annotations'][0]
        
        if 'bbox' in ann:
            return 'detection'
        elif 'segmentation' in ann:
            return 'segmentation'
        elif 'keypoints' in ann:
            return 'pose'
        
        return 'detection'  # Default
    
    @staticmethod
    def _determine_labelme_type(shapes: list) -> str:
        """Determine type from LabelMe shapes"""
        if not shapes:
            return 'classification'
        
        shape_types = set(shape.get('shape_type', '') for shape in shapes)
        
        if 'polygon' in shape_types or 'mask' in shape_types:
            return 'segmentation'
        elif 'rectangle' in shape_types or 'line' in shape_types:
            return 'detection'
        elif 'point' in shape_types:
            return 'pose'
        
        return 'detection'
    
    def _extract_class_names(self, root_dir: str, annotation_format: str, annotations: list) -> Dict[int, str]:
        """
        Extract class names from YAML (YOLO), JSON (COCO), or loose JSON maps (ImageNet).
        Returns dict mapping class_id (int) -> class_name (str).
        """
        class_names = {}
        
        # 1) Try YOLO YAML files
        yaml_candidates = ['data.yaml', 'data.yml', 'dataset.yaml', 'dataset.yml']
        for fname in yaml_candidates:
            yaml_path = os.path.join(root_dir, fname)
            if os.path.isfile(yaml_path):
                try:
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    if data and 'names' in data:
                        names = data['names']
                        if isinstance(names, list):
                            class_names = {i: name for i, name in enumerate(names)}
                        elif isinstance(names, dict):
                            class_names = {int(k): str(v) for k, v in names.items()}
                    if class_names:
                        return class_names
                except Exception:
                    pass
        
        # 2) Try COCO JSON Categories
        if annotation_format == 'coco' and annotations:
            try:
                for ann_file in annotations:
                    if ann_file.endswith('.json'):
                        with open(ann_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if 'categories' in data:
                                for cat in data['categories']:
                                    if 'id' in cat and 'name' in cat:
                                        class_names[int(cat['id'])] = str(cat['name'])
                                if class_names:
                                    return class_names
            except Exception:
                pass
                
        # 3) Try loose JSON class maps in root directory (e.g. ImageNet idx map)
        try:
            for fname in os.listdir(root_dir):
                if fname.endswith('.json') and 'annotations' not in fname.lower():
                    json_path = os.path.join(root_dir, fname)
                    if os.path.isfile(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # Expecting simple dict like: {"0": "tench", "1": "goldfish"}
                        if isinstance(data, dict):
                            # Verify if keys are numeric
                            numeric_keys = 0
                            for k, v in data.items():
                                if str(k).isdigit() and isinstance(v, str):
                                    class_names[int(k)] = v
                                    numeric_keys += 1
                            if numeric_keys > 0 and numeric_keys >= len(data) * 0.5:
                                return class_names
        except Exception:
            pass

        return class_names
