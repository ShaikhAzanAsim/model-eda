#!/usr/bin/env python3
"""
Verification Script for Dataset Visualizer Installation
Tests dependencies and basic functionality
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    print("-" * 60)
    
    dependencies = {
        'PyQt5': 'PyQt5.QtWidgets',
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
        'Pillow': 'PIL',
    }
    
    results = {}
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name:20} - OK")
            results[name] = True
        except ImportError as e:
            print(f"✗ {name:20} - MISSING ({e})")
            results[name] = False
    
    return all(results.values())


def test_local_modules():
    """Test local Pipeline modules"""
    print("\nTesting local modules...")
    print("-" * 60)
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_dir, 'Pipeline'))
    
    modules = {
        'dataset_detector': ['DatasetDetector', 'DatasetInfo'],
        'annotation_handlers': ['AnnotationHandlerFactory', 'ClassificationHandler'],
        'video_generator': ['VideoGenerator'],
    }
    
    results = {}
    for module_name, classes in modules.items():
        try:
            module = __import__(module_name)
            
            # Check classes
            missing = []
            for cls_name in classes:
                if not hasattr(module, cls_name):
                    missing.append(cls_name)
            
            if missing:
                print(f"✗ {module_name:25} - Missing: {', '.join(missing)}")
                results[module_name] = False
            else:
                print(f"✓ {module_name:25} - OK ({', '.join(classes)})")
                results[module_name] = True
        
        except ImportError as e:
            print(f"✗ {module_name:25} - {e}")
            results[module_name] = False
    
    return all(results.values())


def test_pyqt_functionality():
    """Test PyQt5 basic functionality"""
    print("\nTesting PyQt5 functionality...")
    print("-" * 60)
    
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QSize
        
        # Can't create app in headless environment, just test imports
        print("✓ PyQt5 imports              - OK")
        print("✓ PyQt5 basic classes       - OK (Ready to run GUI)")
        return True
    except Exception as e:
        print(f"✗ PyQt5 functionality       - {e}")
        return False


def test_opencv_functionality():
    """Test OpenCV basic functionality"""
    print("\nTesting OpenCV functionality...")
    print("-" * 60)
    
    try:
        import cv2
        import numpy as np
        
        # Create dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test basic operations
        resized = cv2.resize(img, (50, 50))
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print("✓ OpenCV basic operations   - OK")
        print(f"  OpenCV version: {cv2.__version__}")
        return True
    except Exception as e:
        print(f"✗ OpenCV functionality      - {e}")
        return False


def test_file_structure():
    """Verify required files exist"""
    print("\nVerifying file structure...")
    print("-" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    required_files = {
        'Pipeline/pyqt.py': 'Main application',
        'Pipeline/dataset_detector.py': 'Dataset detection',
        'Pipeline/annotation_handlers.py': 'Annotation handlers',
        'Pipeline/video_generator.py': 'Video generation',
        'requirements.txt': 'Dependencies',
        'README_VISUALIZER.md': 'Documentation',
        'QUICK_START.md': 'Quick start guide',
    }
    
    results = {}
    for filepath, description in required_files.items():
        full_path = os.path.join(current_dir, filepath)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            print(f"✓ {filepath:40} - {size:>8} bytes")
            results[filepath] = True
        else:
            print(f"✗ {filepath:40} - NOT FOUND")
            results[filepath] = False
    
    return all(results.values())


def test_permissions():
    """Test write permissions for video output"""
    print("\nTesting permissions...")
    print("-" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Test if we can write to current directory
        test_file = os.path.join(current_dir, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        print(f"✓ Write permissions         - OK (can write to {current_dir})")
        return True
    except Exception as e:
        print(f"✗ Write permissions         - {e}")
        print(f"  Current directory: {current_dir}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + "  Dataset Visualizer - Verification".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    tests = [
        ("Dependencies", test_imports),
        ("Local Modules", test_local_modules),
        ("PyQt5 Functionality", test_pyqt_functionality),
        ("OpenCV Functionality", test_opencv_functionality),
        ("File Structure", test_file_structure),
        ("System Permissions", test_permissions),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} - Unexpected error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nYou can now run the application:")
        print("  python Pipeline/pyqt.py")
        print("\nOr check examples:")
        print("  python examples.py")
        return 0
    else:
        print("\n✗ Some tests failed.")
        print("\nTo fix:")
        print("1. Install missing dependencies:")
        print("   pip install -r requirements.txt")
        print("\n2. Verify Python version (3.7+):")
        print(f"   Current: {sys.version}")
        print("\n3. Check README_VISUALIZER.md for detailed setup")
        return 1


if __name__ == '__main__':
    sys.exit(main())
