"""
Utility functions for the breed detection project
"""

import os
import shutil
from pathlib import Path
import json


def create_sample_directory_structure(base_dir='data'):
    """
    Create sample directory structure for organizing dataset
    """
    data_dir = Path(base_dir) / 'images'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory: {data_dir}")
    print("\nPlease organize your images in the following structure:")
    print(f"{data_dir}/")
    print("├── breed_1/")
    print("│   ├── image1.jpg")
    print("│   ├── image2.jpg")
    print("│   └── ...")
    print("├── breed_2/")
    print("│   ├── image1.jpg")
    print("│   └── ...")
    print("└── ...")
    print("\nEach breed folder should contain images of that specific breed.")


def check_dataset_structure(data_dir='data/images'):
    """
    Check if dataset is properly organized
    Returns: dict with statistics
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return None
    
    stats = {
        'total_breeds': 0,
        'total_images': 0,
        'breeds': {}
    }
    
    for breed_folder in sorted(data_path.iterdir()):
        if breed_folder.is_dir():
            breed_name = breed_folder.name
            image_files = list(breed_folder.glob('*.jpg')) + \
                         list(breed_folder.glob('*.jpeg')) + \
                         list(breed_folder.glob('*.png'))
            
            stats['breeds'][breed_name] = len(image_files)
            stats['total_images'] += len(image_files)
            stats['total_breeds'] += 1
    
    return stats


def print_dataset_statistics(data_dir='data/images'):
    """Print dataset statistics"""
    stats = check_dataset_structure(data_dir)
    
    if stats is None:
        return
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total Breeds: {stats['total_breeds']}")
    print(f"Total Images: {stats['total_images']}")
    print(f"Average Images per Breed: {stats['total_images'] / stats['total_breeds']:.1f}")
    print("\nImages per Breed:")
    print("-"*70)
    
    for breed, count in sorted(stats['breeds'].items(), key=lambda x: x[1], reverse=True):
        print(f"{breed:30s}: {count:4d} images")
    
    print("="*70)


def save_dataset_info(data_dir='data/images', output_file='results/dataset_info.json'):
    """Save dataset information to JSON file"""
    stats = check_dataset_structure(data_dir)
    
    if stats is None:
        return
    
    os.makedirs('results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset information saved to {output_file}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'create':
            create_sample_directory_structure()
        elif sys.argv[1] == 'check':
            data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/images'
            print_dataset_statistics(data_dir)
        elif sys.argv[1] == 'save':
            data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/images'
            save_dataset_info(data_dir)
    else:
        print("Usage:")
        print("  python utils.py create              - Create directory structure")
        print("  python utils.py check [data_dir]    - Check dataset statistics")
        print("  python utils.py save [data_dir]     - Save dataset info to JSON")

