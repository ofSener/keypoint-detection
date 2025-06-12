#!/usr/bin/env python3
"""
preprocess_sift.py - SIFT keypoint ve descriptor'ları önceden hesapla
Bu script tüm görüntüler için SIFT'i hesaplayıp .npz dosyalarına kaydeder
Training sırasında çok daha hızlı olur.
"""

import argparse
import pathlib
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def process_single_image(args):
    """Tek bir görüntü için SIFT hesapla ve kaydet"""
    img_path, output_dir, max_kp = args
    
    try:
        # Output path
        rel_path = img_path.relative_to(img_path.parents[2])  # Remove base path
        npz_path = output_dir / rel_path.with_suffix('.npz')
        
        # Skip if already exists
        if npz_path.exists():
            return f"Skip: {rel_path}"
        
        # Create output directory
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load and resize image for faster processing
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return f"Error: Could not load {rel_path}"
        
        # Resize to smaller size for faster SIFT
        h, w = img_bgr.shape[:2]
        scale = min(1.0, 640 / max(h, w))  # Max dimension 640px
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h))
            
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Faster SIFT parameters
        sift = cv2.SIFT_create(
            nfeatures=max_kp,
            nOctaveLayers=3,  # Default is 3
            contrastThreshold=0.04,  # Default is 0.04
            edgeThreshold=10,  # Default is 10
            sigma=1.6  # Default is 1.6
        )
        
        # Compute SIFT with faster parameters
        kps, des = sift.detectAndCompute(gray, None)
        
        if des is None:
            xy = np.empty((0, 2), np.int32)
            des = np.empty((0, 128), np.float32)
        else:
            kps = kps[:max_kp]
            des = des[:max_kp]
            xy = np.round([[kp.pt[0]/scale, kp.pt[1]/scale] for kp in kps]).astype(np.int32)  # Scale back to original size
            des = des.astype(np.float32) / 512.0  # Normalize
        
        # Save to .npz with compression level 1 (fastest)
        np.savez(npz_path, xy=xy, d=des)  # Removed compression for speed
        return f"Done: {rel_path}"
        
    except Exception as e:
        return f"Error {rel_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='~/projects/keypoint/data/processed', 
                       help='Path to processed data directory')
    parser.add_argument('--max_kp', type=int, default=300,  # Reduced from 500
                       help='Maximum keypoints per image')
    parser.add_argument('--workers', type=int, default=cpu_count(), 
                       help='Number of worker processes')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Limit number of images (for testing)')
    parser.add_argument('--chunk_size', type=int, default=32,
                       help='Chunk size for multiprocessing')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = pathlib.Path(args.data).expanduser()
    output_dir = data_dir.parent / "kp_desc"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Max keypoints: {args.max_kp}")
    print(f"Chunk size: {args.chunk_size}")
    
    # Find all PNG files
    png_files = sorted(data_dir.glob("**/*.png"))
    if not png_files:
        print(f"No PNG files found in {data_dir}")
        return
    
    # Limit images if specified
    if args.max_images:
        png_files = png_files[:args.max_images]
        print(f"Limited to {len(png_files)} images")
    
    print(f"Found {len(png_files)} images")
    
    # Check how many already exist
    existing = 0
    for img_path in png_files:
        rel_path = img_path.relative_to(img_path.parents[2])
        npz_path = output_dir / rel_path.with_suffix('.npz')
        if npz_path.exists():
            existing += 1
    
    print(f"Already processed: {existing}/{len(png_files)}")
    print(f"To process: {len(png_files) - existing}")
    
    # Prepare arguments for multiprocessing
    process_args = [(img_path, output_dir, args.max_kp) for img_path in png_files]
    
    # Process with multiprocessing using imap_unordered for better performance
    print("\nStarting SIFT preprocessing...")
    print("(Using faster settings: resized images, fewer keypoints, no compression)")
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_image, process_args, chunksize=args.chunk_size),
            total=len(process_args),
            desc="Processing images"
        ))
    
    # Count results
    done = sum(1 for r in results if r.startswith("Done:"))
    skipped = sum(1 for r in results if r.startswith("Skip:"))
    errors = sum(1 for r in results if r.startswith("Error"))
    
    print(f"\nResults:")
    print(f"  Processed: {done}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total: {len(results)}")
    
    if errors > 0:
        print("\nErrors:")
        for r in results:
            if r.startswith("Error"):
                print(f"  {r}")

if __name__ == "__main__":
    main() 