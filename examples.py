"""
Example: Using the Dataset Visualizer Programmatically

This script demonstrates how to use the visualizer components
without the PyQt5 GUI for batch processing or integration into
other applications.
"""

import os
import sys
import random
from pathlib import Path

# Add Pipeline to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Pipeline'))

from dataset_detector import DatasetDetector
from annotation_handlers import AnnotationHandlerFactory
from video_generator import VideoGenerator


def example_1_detect_dataset():
    """Example 1: Detect dataset type and structure"""
    print("=" * 60)
    print("Example 1: Dataset Detection")
    print("=" * 60)
    
    dataset_path = "/path/to/your/dataset"  # Change this to your dataset
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    detector = DatasetDetector()
    dataset_info = detector.detect_dataset(dataset_path)
    
    if dataset_info:
        print(f"\nDataset Type: {dataset_info.dataset_type}")
        print(f"Total Images: {dataset_info.metadata['total_images']}")
        print(f"Total Annotations: {dataset_info.metadata['total_annotations']}")
        print(f"Annotation Format: {dataset_info.annotation_format}")
        print(f"Separate Directories: {dataset_info.has_separate_dirs}")
        print(f"Image Directory: {dataset_info.image_dir}")
        print(f"Annotation Directory: {dataset_info.annotation_dir}")
        
        if dataset_info.image_paths:
            print(f"\nFirst 3 images:")
            for img in dataset_info.image_paths[:3]:
                print(f"  - {img}")
    else:
        print("No dataset found!")


def example_2_annotate_samples():
    """Example 2: Load and annotate random samples"""
    print("\n" + "=" * 60)
    print("Example 2: Annotate Random Samples")
    print("=" * 60)
    
    dataset_path = "/path/to/your/dataset"  # Change this
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Detect dataset
    detector = DatasetDetector()
    dataset_info = detector.detect_dataset(dataset_path)
    
    if not dataset_info:
        print("No dataset found!")
        return
    
    print(f"\nDetected: {dataset_info.dataset_type}")
    
    # Get handler
    factory = AnnotationHandlerFactory()
    handler = factory.get_handler(dataset_info.dataset_type, dataset_info)
    
    # Process random samples
    num_samples = 5
    sample_indices = random.sample(
        range(len(dataset_info.image_paths)),
        min(num_samples, len(dataset_info.image_paths))
    )
    
    annotated_images = []
    for idx, sample_idx in enumerate(sample_indices):
        image_path = dataset_info.image_paths[sample_idx]
        print(f"Processing {idx + 1}/{num_samples}: {Path(image_path).name}")
        
        try:
            annotated = handler.annotate_image(image_path, grid_cells=3)
            annotated_images.append(annotated)
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nSuccessfully annotated {len(annotated_images)} samples")
    return annotated_images


def example_3_generate_video():
    """Example 3: Generate video from annotated samples"""
    print("\n" + "=" * 60)
    print("Example 3: Generate Slideshow Video")
    print("=" * 60)
    
    dataset_path = "/path/to/your/dataset"  # Change this
    num_samples = 10
    grid_size = 3
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    # Step 1: Detect
    print("Step 1: Detecting dataset...")
    detector = DatasetDetector()
    dataset_info = detector.detect_dataset(dataset_path)
    
    if not dataset_info:
        print("No dataset found!")
        return
    
    print(f"  Detected: {dataset_info.dataset_type}")
    
    # Step 2: Get handler
    print("Step 2: Preparing annotation handler...")
    factory = AnnotationHandlerFactory()
    handler = factory.get_handler(dataset_info.dataset_type, dataset_info)
    
    # Step 3: Sample and annotate
    print(f"Step 3: Processing {num_samples} random samples...")
    sample_indices = random.sample(
        range(len(dataset_info.image_paths)),
        min(num_samples, len(dataset_info.image_paths))
    )
    
    frames = []
    for idx, sample_idx in enumerate(sample_indices):
        image_path = dataset_info.image_paths[sample_idx]
        
        try:
            annotated = handler.annotate_image(image_path, grid_cells=grid_size)
            # Duplicate for 2 seconds at 30fps
            frames.extend([annotated] * 60)
            
            if (idx + 1) % 5 == 0:
                print(f"  Processed {idx + 1}/{num_samples}")
        except Exception as e:
            print(f"  Error processing {Path(image_path).name}: {e}")
    
    # Step 4: Generate video
    print("Step 4: Generating video...")
    video_gen = VideoGenerator(fps=30, codec='mp4v')
    
    output_path = os.path.join(dataset_path, "slideshow_custom.mp4")
    success = video_gen.create_video(frames, output_path)
    
    if success:
        print(f"\n✓ Video generated successfully!")
        print(f"  Location: {output_path}")
        print(f"  Frames: {len(frames)}")
        print(f"  Duration: {len(frames) / 30:.1f} seconds")
    else:
        print("\n✗ Video generation failed")


def example_4_custom_handler():
    """Example 4: Create and register a custom annotation handler"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Annotation Handler")
    print("=" * 60)
    
    from annotation_handlers import AnnotationHandler
    import cv2
    
    class CustomDatasetHandler(AnnotationHandler):
        """Custom handler for specialized dataset"""
        
        def annotate_image(self, image_path: str, grid_cells: int):
            img = self._load_image(image_path).copy()
            
            # Your custom annotation logic here
            # Example: Add custom text and draw custom shapes
            h, w = img.shape[:2]
            
            # Add custom text
            cv2.putText(img, "Custom Processing", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw custom shape
            cv2.circle(img, (w // 2, h // 2), 50, (255, 0, 0), 2)
            
            img = cv2.resize(img, (512, 512))
            return img
    
    # Register custom handler
    factory = AnnotationHandlerFactory()
    factory.register_handler('custom_type', CustomDatasetHandler)
    
    print("✓ Custom handler registered for 'custom_type' datasets")
    
    # Retrieve handler
    dataset_info = type('obj', (object,), {'image_dir': '.', 'annotation_dir': '.'})()
    handler = factory.get_handler('custom_type', dataset_info)
    print(f"✓ Handler retrieved: {type(handler).__name__}")


def example_5_batch_processing():
    """Example 5: Process multiple datasets in batch"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing Multiple Datasets")
    print("=" * 60)
    
    dataset_dirs = [
        "/path/to/dataset1",
        "/path/to/dataset2",
        "/path/to/dataset3",
    ]
    
    detector = DatasetDetector()
    factory = AnnotationHandlerFactory()
    video_gen = VideoGenerator()
    
    results = []
    
    for dataset_path in dataset_dirs:
        if not os.path.exists(dataset_path):
            print(f"Skipping: {dataset_path} (not found)")
            continue
        
        print(f"\nProcessing: {dataset_path}")
        
        try:
            # Detect
            dataset_info = detector.detect_dataset(dataset_path)
            if not dataset_info:
                print("  No dataset detected")
                continue
            
            print(f"  Type: {dataset_info.dataset_type}")
            print(f"  Images: {dataset_info.metadata['total_images']}")
            
            results.append({
                'path': dataset_path,
                'type': dataset_info.dataset_type,
                'images': dataset_info.metadata['total_images'],
                'status': 'success'
            })
        
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'path': dataset_path,
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary")
    print("=" * 60)
    for result in results:
        if result['status'] == 'success':
            print(f"✓ {Path(result['path']).name}")
            print(f"    Type: {result['type']}, Images: {result['images']}")
        else:
            print(f"✗ {Path(result['path']).name}")
            print(f"    Error: {result.get('error', 'Unknown')}")


if __name__ == '__main__':
    """
    Run examples to demonstrate programmatic usage.
    
    Before running, update the dataset paths in each example
    to point to your actual datasets.
    """
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Dataset Visualizer - Usage Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    # Uncomment to run:
    
    # example_1_detect_dataset()          # Basic detection
    # example_2_annotate_samples()        # Annotate samples
    # example_3_generate_video()          # Create video
    # example_4_custom_handler()          # Custom handler
    # example_5_batch_processing()        # Batch mode
    
    print("\n" + "=" * 60)
    print("Examples ready to run!")
    print("=" * 60)
    print("\nTo run examples:")
    print("1. Edit dataset paths in each example function")
    print("2. Uncomment the example calls at the bottom")
    print("3. Run this script: python examples.py")
    print("\nRefer to README_VISUALIZER.md for more information")
    print("=" * 60)
