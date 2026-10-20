"""
PyQt5 Application for Image Dataset Visualization and Analysis
Supports multiple dataset types: Classification, Segmentation, Object Detection, Pose Estimation
Auto-detects dataset type and directory structure
Generates slideshow video of random samples in a grid
"""

import sys
import os
import csv
import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import traceback

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QSpinBox, QDoubleSpinBox, QComboBox, QProgressBar,
    QMessageBox, QScrollArea, QFrame, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw, ImageFont

from dataset_detector import DatasetDetector, DatasetInfo
from annotation_handlers import AnnotationHandlerFactory
from video_generator import VideoGenerator


def _extract_annotation_info(image_path: str, dataset_info, handler) -> dict:
    """
    Extract annotation metadata for a single sample.
    Returns a dict mapping class/label name -> occurrence count.
    Works for: classification, detection, segmentation, pose, video.
    """
    dataset_type = dataset_info.dataset_type.lower()
    class_names = getattr(dataset_info, 'class_names', {})

    def resolve(raw):
        if isinstance(raw, int) and raw in class_names:
            return str(class_names[raw])
        return str(raw)

    # ── Classification / Video ────────────────────────────────────────────
    if dataset_type in ('classification', 'unknown'):
        parent = Path(image_path).parent.name
        if parent.isdigit() and class_names:
            label = class_names.get(int(parent), parent)
        else:
            label = parent or 'Unknown'
        return {label: 1}

    # ── Detection / Segmentation ─────────────────────────────────────────
    if dataset_type in ('detection', 'segmentation'):
        counts = {}

        # 1) YOLO .txt
        txt = handler._find_annotation_for_image(image_path, '.txt')
        if txt:
            try:
                with open(txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            name = resolve(int(parts[0]))
                            counts[name] = counts.get(name, 0) + 1
            except Exception:
                pass
            if counts:
                return counts

        # 2) COCO global cache (pre-built by the handler)
        if getattr(dataset_info, 'annotation_format', '') == 'coco':
            fname = Path(image_path).name
            for ann in handler.coco_image_cache.get(fname, []):
                name = resolve(ann.get('category_id', 'Unknown'))
                counts[name] = counts.get(name, 0) + 1
            if counts:
                return counts

        # 3) JSON (LabelMe or local COCO)
        json_f = handler._find_annotation_for_image(image_path, '.json')
        if json_f:
            try:
                with open(json_f, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'shapes' in data:          # LabelMe
                    for shape in data['shapes']:
                        lbl = str(shape.get('label', 'Unknown'))
                        counts[lbl] = counts.get(lbl, 0) + 1
                elif 'annotations' in data:   # COCO subset
                    for ann in data['annotations']:
                        name = resolve(ann.get('category_id', 'Unknown'))
                        counts[name] = counts.get(name, 0) + 1
            except Exception:
                pass
        return counts if counts else {'Unknown': 0}

    # ── Pose ─────────────────────────────────────────────────────────────
    if dataset_type == 'pose':
        count = 0
        json_f = handler._find_annotation_for_image(image_path, '.json')
        if json_f:
            try:
                with open(json_f, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'annotations' in data:
                    count = sum(1 for a in data['annotations'] if 'keypoints' in a)
                elif 'keypoints' in data:
                    count = 1
            except Exception:
                pass
        return {'persons': count or 1}

    # Fallback
    parent = Path(image_path).parent.name
    return {parent: 1} if parent else {'Unknown': 1}


class VisualizationWorker(QThread):
    """Worker thread for processing dataset visualization"""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, dataset_path: str, n_samples: int, grid_size: int,
                 slide_duration: float = 2.0, duration_mode: str = 'manual'):
        super().__init__()
        self.dataset_path = dataset_path
        self.n_samples = n_samples
        self.grid_size = grid_size
        self.slide_duration = slide_duration
        self.duration_mode = duration_mode  # 'manual' or 'auto_video'
        self.detector = DatasetDetector()
        self.handler_factory = AnnotationHandlerFactory()
        self.video_gen = VideoGenerator()
        
    def run(self):
        try:
            self.status.emit("Detecting dataset type...")
            self.progress.emit(10)
            
            # Detect dataset information
            dataset_info = self.detector.detect_dataset(self.dataset_path)
            if not dataset_info or not dataset_info.image_paths:
                self.finished.emit(False, "No images found in the dataset")
                return
            
            self.status.emit(f"Detected: {dataset_info.dataset_type}")
            self.progress.emit(20)
            
            # Get annotation handler
            handler = self.handler_factory.get_handler(
                dataset_info.dataset_type,
                dataset_info
            )
            
            self.status.emit("Loading random samples...")
            self.progress.emit(30)

            # Random sample selection
            sample_indices = random.sample(
                range(len(dataset_info.image_paths)),
                min(self.n_samples, len(dataset_info.image_paths))
            )

            # Grid grouping constant needed before annotation
            cells_per_grid = self.grid_size * self.grid_size
            video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            VIDEO_FPS = 30  # extraction / output fps

            # ── Pre-scan video durations ──────────────────────────────────
            # Build a dict: path -> duration in seconds (videos only)
            vid_durations: Dict[str, float] = {}
            for si in sample_indices:
                ip = dataset_info.image_paths[si]
                if Path(ip).suffix.lower() in video_exts:
                    cap = cv2.VideoCapture(ip)
                    if cap.isOpened():
                        fps_v = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
                        fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        vid_durations[ip] = (fc / fps_v) if fc > 0 else self.slide_duration

            # Per-grid minimum video duration (used for both clip length and slide timing)
            grid_min_dur: Dict[int, float] = {}
            for g_idx, start in enumerate(range(0, len(sample_indices), cells_per_grid)):
                group_paths = [
                    dataset_info.image_paths[si]
                    for si in sample_indices[start:start + cells_per_grid]
                ]
                vid_paths_in_group = [p for p in group_paths if Path(p).suffix.lower() in video_exts]
                if vid_paths_in_group and self.duration_mode == 'auto_video':
                    min_dur = min(vid_durations.get(p, self.slide_duration) for p in vid_paths_in_group)
                else:
                    min_dur = self.slide_duration
                grid_min_dur[g_idx] = min_dur

            # ── Annotate samples ─────────────────────────────────────────
            # For video files num_frames controls clip length = duration * fps
            annotated_frames = []
            sampled_paths = []  # parallel to annotated_frames (only successful samples)
            for idx, sample_idx in enumerate(sample_indices):
                try:
                    image_path = dataset_info.image_paths[sample_idx]
                    grid_idx = idx // cells_per_grid
                    clip_dur = grid_min_dur.get(grid_idx, self.slide_duration)
                    # Cap extraction to avoid RAM exhaustion; VideoGenerator will
                    # repeat frames to fill the full target duration.
                    MAX_FRAMES_PER_CLIP = 150
                    num_frames = max(1, min(int(round(clip_dur * VIDEO_FPS)),
                                           MAX_FRAMES_PER_CLIP))
                    frame = handler.annotate_image(image_path, self.grid_size,
                                                   num_frames=num_frames)
                    annotated_frames.append(frame)
                    sampled_paths.append(image_path)
                except Exception as e:
                    print(f"Error processing sample {sample_idx}: {e}")
                    continue

                progress_val = 30 + (50 * (idx + 1) // len(sample_indices))
                self.progress.emit(progress_val)

            if not annotated_frames:
                self.finished.emit(False, "Failed to process any samples")
                return

            self.status.emit("Building grid frames...")
            self.progress.emit(80)

            # Group annotated frames into batches of grid_size * grid_size
            image_groups = []
            for i in range(0, len(annotated_frames), cells_per_grid):
                group = annotated_frames[i:i + cells_per_grid]
                image_groups.append(group)

            self.status.emit("Generating grid video...")
            self.progress.emit(85)

            # Per-grid slide durations (used for static-image grids; video grids
            # are driven by num_frames extracted above)
            per_grid_durations = [
                grid_min_dur.get(g_idx, self.slide_duration)
                for g_idx in range(len(image_groups))
            ]

            # Generate video using grid layout
            output_path = os.path.join(
                self.dataset_path,
                "slideshow_video.mp4"
            )
            success = self.video_gen.create_grid_video(
                image_groups, self.grid_size, output_path,
                slide_durations=per_grid_durations
            )
            
            if success:
                self.progress.emit(100)
                result_msg = f"Video saved successfully at:\n{output_path}"

                # Write CSV index file for every slideshow regardless of dataset type
                self.status.emit("Writing sample index file...")

                # Collect per-grid sample data first
                # grid_data[grid_idx] = list of (title, annotation_info) per sample
                grid_data: Dict[int, list] = {}
                for grid_idx, i in enumerate(range(0, len(sampled_paths), cells_per_grid)):
                    grid_samples = sampled_paths[i:i + cells_per_grid]
                    entries = []
                    for sample_path in grid_samples:
                        sample_title = Path(sample_path).stem
                        annotation_info = _extract_annotation_info(
                            sample_path, dataset_info, handler
                        )
                        # Format annotation dict as "class1:count1, class2:count2"
                        ann_str = ", ".join(
                            f"{k}:{v}" for k, v in annotation_info.items()
                        )
                        entries.append(f"{sample_title} | {ann_str}")
                    grid_data[grid_idx] = entries

                # CSV layout: rows = samples, columns = grids
                num_grids = len(grid_data)
                max_samples = max((len(v) for v in grid_data.values()), default=0)
                grid_headers = [f"grid_{g}" for g in range(num_grids)]

                index_path = os.path.join(self.dataset_path, "slideshow_index.csv")
                with open(index_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["sample"] + grid_headers)
                    for row_idx in range(max_samples):
                        row = [f"sample_{row_idx}"]
                        for g_idx in range(num_grids):
                            entries = grid_data.get(g_idx, [])
                            row.append(entries[row_idx] if row_idx < len(entries) else "")
                        writer.writerow(row)
                result_msg += f"\nIndex saved at:\n{index_path}"

                self.finished.emit(True, result_msg)
            else:
                self.finished.emit(False, "Failed to generate video")
                
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}\n{traceback.format_exc()}")


class DatasetVisualizer(QWidget):
    """Main PyQt application for dataset visualization"""
    
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Image Dataset Visualizer & Analyzer")
        self.setGeometry(100, 100, 700, 400)
        
        # Directory input
        dir_label = QLabel("Dataset Directory:")
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Enter dataset directory path...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_directory)
        
        # Sample size input
        sample_label = QLabel("Number of Random Samples (N):")
        self.sample_spinbox = QSpinBox()
        self.sample_spinbox.setMinimum(1)
        self.sample_spinbox.setMaximum(1000)
        self.sample_spinbox.setValue(10)
        
        # Grid size input
        grid_label = QLabel("Grid Size:")
        self.grid_combobox = QComboBox()
        self.grid_combobox.addItems(["2x2", "3x3", "4x4", "5x5", "6x6"])
        self.grid_combobox.setCurrentText("3x3")
        
        # Slide duration mode (video datasets)
        duration_mode_label = QLabel("Slide Duration Mode (video datasets):")
        self.radio_manual = QRadioButton("Manual (set below)")
        self.radio_auto = QRadioButton("Auto – shortest video per grid")
        self.radio_manual.setChecked(True)
        self.duration_mode_group = QButtonGroup(self)
        self.duration_mode_group.addButton(self.radio_manual)
        self.duration_mode_group.addButton(self.radio_auto)
        self.radio_manual.toggled.connect(self._on_duration_mode_changed)

        # Slide duration spinbox
        duration_label = QLabel("Slide Duration (seconds):")
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setMinimum(0.5)
        self.duration_spinbox.setMaximum(5.0)
        self.duration_spinbox.setSingleStep(0.5)
        self.duration_spinbox.setValue(2.0)
        self.duration_spinbox.setDecimals(1)
        self.duration_spinbox.setSuffix(" s")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: blue;")
        
        # Process button
        self.process_btn = QPushButton("Generate Slideshow")
        self.process_btn.clicked.connect(self.start_processing)
        
        # Layouts
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.browse_btn)
        
        sample_layout = QHBoxLayout()
        sample_layout.addWidget(sample_label)
        sample_layout.addWidget(self.sample_spinbox)
        sample_layout.addStretch()
        
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(grid_label)
        grid_layout.addWidget(self.grid_combobox)
        grid_layout.addStretch()
        
        duration_mode_layout = QHBoxLayout()
        duration_mode_layout.addWidget(duration_mode_label)
        duration_mode_layout.addWidget(self.radio_manual)
        duration_mode_layout.addWidget(self.radio_auto)
        duration_mode_layout.addStretch()

        duration_layout = QHBoxLayout()
        duration_layout.addWidget(duration_label)
        duration_layout.addWidget(self.duration_spinbox)
        duration_layout.addStretch()
        
        main_layout = QVBoxLayout()
        main_layout.addWidget(dir_label)
        main_layout.addLayout(dir_layout)
        main_layout.addLayout(sample_layout)
        main_layout.addLayout(grid_layout)
        main_layout.addLayout(duration_mode_layout)
        main_layout.addLayout(duration_layout)
        main_layout.addWidget(QLabel("Progress:"))
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.process_btn)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def _on_duration_mode_changed(self):
        """Enable/disable the duration spinbox based on selected mode"""
        self.duration_spinbox.setEnabled(self.radio_manual.isChecked())

    def browse_directory(self):
        """Open directory selection dialog"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Dataset Directory"
        )
        if directory:
            self.dir_input.setText(directory)
    
    def start_processing(self):
        """Start the visualization processing"""
        dataset_path = self.dir_input.text().strip()
        
        if not dataset_path:
            QMessageBox.warning(self, "Input Error", "Please select a dataset directory")
            return
        
        if not os.path.isdir(dataset_path):
            QMessageBox.warning(self, "Path Error", "Invalid directory path")
            return
        
        n_samples = self.sample_spinbox.value()
        grid_size = int(self.grid_combobox.currentText().split('x')[0])
        slide_duration = self.duration_spinbox.value()
        duration_mode = 'manual' if self.radio_manual.isChecked() else 'auto_video'

        # Disable button and reset progress
        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        self.status_label.setStyleSheet("color: orange;")

        # Start worker thread
        self.worker = VisualizationWorker(
            dataset_path, n_samples, grid_size, slide_duration, duration_mode
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
    
    def update_progress(self, value: int):
        """Update progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(message)
    
    def on_finished(self, success: bool, message: str):
        """Handle completion of processing"""
        self.process_btn.setEnabled(True)
        
        if success:
            self.status_label.setStyleSheet("color: green;")
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "Error", message)
        
        self.status_label.setText("Ready" if success else "Failed")


def main():
    app = QApplication(sys.argv)
    visualizer = DatasetVisualizer()
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()