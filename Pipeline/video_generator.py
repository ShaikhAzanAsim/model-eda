"""
Video Generator Module
Creates slideshow video from grid of annotated images
"""

import os
import cv2
import numpy as np
from typing import List


class VideoGenerator:
    """Generate MP4 video from image frames"""
    
    def __init__(self, fps: int = 30, codec: str = 'mp4v'):
        self.fps = fps
        self.codec = codec
    
    def create_video(self, frames: List[np.ndarray], output_path: str) -> bool:
        """
        Create video from list of frames
        
        Args:
            frames: List of numpy arrays (BGR images)
            output_path: Path to save video file
        
        Returns:
            True if successful, False otherwise
        """
        if not frames:
            print("No frames provided")
            return False
        
        try:
            # Get frame dimensions from first frame
            h, w = frames[0].shape[:2]
            
            # Define codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
            
            if not out.isOpened():
                print(f"Failed to open video writer for {output_path}")
                return False
            
            # Write frames
            for i, frame in enumerate(frames):
                # Ensure frame has correct shape
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                
                # Ensure frame is BGR
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                out.write(frame)
                
                if (i + 1) % max(1, len(frames) // 10) == 0:
                    print(f"Written {i + 1}/{len(frames)} frames")
            
            out.release()
            
            # Verify file was created
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Video successfully created: {output_path}")
                print(f"Video duration: {len(frames) / self.fps:.2f} seconds")
                return True
            else:
                print("Video file was not created properly")
                return False
        
        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return False
    
    def create_grid_video(self, image_groups: List[List[np.ndarray]],
                         grid_size: int, output_path: str,
                         slide_duration: float = 2.0,
                         slide_durations: List[float] = None) -> bool:
        """
        Create video from grouped images arranged in grid
        
        Args:
            image_groups: List of lists, each inner list contains grid_size*grid_size images
            grid_size: Size of grid (2 for 2x2, 3 for 3x3, etc.)
            output_path: Path to save video
            slide_duration: Uniform duration (seconds) for every grid slide (default 2.0)
            slide_durations: Optional per-grid list of durations (seconds). When provided,
                             overrides slide_duration for each grid individually.
        
        Returns:
            True if successful, False otherwise
        """
        if not image_groups or not image_groups[0]:
            print("No images provided")
            return False
        
        try:
            frames = []

            for group_idx, image_list in enumerate(image_groups):
                # Determine duration for this specific grid
                if slide_durations and group_idx < len(slide_durations):
                    dur = slide_durations[group_idx]
                else:
                    dur = slide_duration
                frames_per_slide = max(1, int(round(self.fps * dur)))
                # Find maximum length of sequence in this group
                max_frames = 1
                for item in image_list:
                    if isinstance(item, list) and len(item) > max_frames:
                        max_frames = len(item)
                
                # If everything is static, duplicate a single generated frame
                if max_frames == 1:
                    grid_frame = self._create_grid_frame(image_list, grid_size)
                    frames.extend([grid_frame] * frames_per_slide)
                else:
                    # Repeat each extracted frame proportionally so the grid
                    # fills the full target duration even when extraction was
                    # capped below frames_per_slide.
                    repeat_per_frame = max(1, frames_per_slide // max_frames)
                    written = 0
                    for t in range(max_frames):
                        current_images = []
                        for item in image_list:
                            if isinstance(item, list):
                                current_images.append(item[min(t, len(item) - 1)])
                            else:
                                current_images.append(item)
                        grid_frame = self._create_grid_frame(current_images, grid_size)
                        for _ in range(repeat_per_frame):
                            frames.append(grid_frame)
                        written += repeat_per_frame
                    # Pad any remaining frames due to integer division rounding
                    shortfall = frames_per_slide - written
                    if shortfall > 0 and frames:
                        for _ in range(shortfall):
                            frames.append(frames[-1])
            
            return self.create_video(frames, output_path)
        
        except Exception as e:
            print(f"Error creating grid video: {str(e)}")
            return False
    
    @staticmethod
    def _create_grid_frame(images: List[np.ndarray], grid_size: int) -> np.ndarray:
        """
        Create a single frame with images arranged in grid
        
        Args:
            images: List of images
            grid_size: Grid dimension (2 for 2x2, etc.)
        
        Returns:
            Combined grid image
        """
        if not images:
            return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Calculate number of rows needed
        num_images = len(images)
        num_cols = grid_size
        num_rows = (num_images + num_cols - 1) // num_cols
        
        # Get cell size from first image
        img_h, img_w = images[0].shape[:2]
        
        # Create canvas
        canvas_h = num_rows * img_h
        canvas_w = num_cols * img_w
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // num_cols
            col = idx % num_cols
            
            y1 = row * img_h
            y2 = y1 + img_h
            x1 = col * img_w
            x2 = x1 + img_w
            
            # Resize image if needed
            if img.shape[:2] != (img_h, img_w):
                img = cv2.resize(img, (img_w, img_h))
            
            # Ensure 3-channel
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            canvas[y1:y2, x1:x2] = img
        
        return canvas
    
    @staticmethod
    def create_montage(images: List[np.ndarray], grid_size: int, 
                       output_path: str, spacing: int = 5) -> bool:
        """
        Create static montage image from grid of images
        
        Args:
            images: List of images
            grid_size: Grid dimension
            output_path: Path to save montage
            spacing: Pixel spacing between images
        
        Returns:
            True if successful
        """
        try:
            grid_frame = VideoGenerator._create_grid_frame(images, grid_size)
            cv2.imwrite(output_path, grid_frame)
            return True
        except Exception as e:
            print(f"Error creating montage: {str(e)}")
            return False
