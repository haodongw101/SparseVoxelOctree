#!/usr/bin/env python3
"""
End-to-end point cloud compression using Sparse Voxel Octree.

This script:
1. Reads a PLY file with voxelized points
2. Converts to voxel fragment format
3. Builds sparse voxel octree using the C++ compressor
4. Optionally saves the compressed octree
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def get_gpu_stats():
    """Get GPU utilization and memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_stats = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_stats.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_total': int(parts[4])
                    })
            return gpu_stats
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def find_compressor():
    """Find the octree_compressor executable."""
    # Try common build directories
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    possible_paths = [
        repo_root / 'build' / 'octree_compressor',
        repo_root / 'build' / 'Release' / 'octree_compressor',
        repo_root / 'build' / 'Debug' / 'octree_compressor',
        repo_root / 'octree_compressor',
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def run_conversion(input_ply, output_fragments, octree_level, no_dedup=False):
    """Run the PLY to fragments conversion."""
    script_dir = Path(__file__).parent
    converter_script = script_dir / 'ply_to_fragments.py'

    cmd = [
        sys.executable,
        str(converter_script),
        '-i', input_ply,
        '-o', output_fragments,
        '-lvl', str(octree_level),
    ]

    if no_dedup:
        cmd.append('--no-dedup')

    print("=" * 60)
    print("Step 1: Converting PLY to voxel fragments")
    print("=" * 60)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nError: PLY conversion failed", file=sys.stderr)
        return False

    return True


def run_compression(fragment_file, octree_level, output_octree=None):
    """Run the octree compression."""
    compressor = find_compressor()

    if not compressor:
        print("\nError: octree_compressor executable not found!", file=sys.stderr)
        print("Please build the project first:", file=sys.stderr)
        print("  mkdir build && cd build", file=sys.stderr)
        print("  cmake .. -DCMAKE_BUILD_TYPE=Release", file=sys.stderr)
        print("  make octree_compressor", file=sys.stderr)
        return False

    print("\n" + "=" * 60)
    print("Step 2: Building octree and compressing")
    print("=" * 60)
    print(f"Using compressor: {compressor}\n")

    cmd = [
        compressor,
        '-i', fragment_file,
        '-lvl', str(octree_level),
    ]

    if output_octree:
        cmd.extend(['-o', output_octree])

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\nError: Octree compression failed", file=sys.stderr)
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Compress point cloud using Sparse Voxel Octree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress a PLY file at level 10
  %(prog)s -i pointcloud.ply -lvl 10

  # Compress and save octree
  %(prog)s -i pointcloud.ply -lvl 10 -o compressed.octree

  # Keep intermediate fragment file
  %(prog)s -i pointcloud.ply -lvl 10 --keep-fragments
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input PLY file with point cloud')
    parser.add_argument('-lvl', '--level', type=int, required=True,
                        help='Octree level (1-12, determines resolution 2^level)')
    parser.add_argument('-o', '--output', help='Output octree file (optional)')
    parser.add_argument('--keep-fragments', action='store_true',
                        help='Keep intermediate fragment file')
    parser.add_argument('--fragment-file', help='Use/save fragment file at this path')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Skip duplicate voxel removal (faster but larger)')

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.level < 1 or args.level > 12:
        print(f"Error: Octree level must be between 1 and 12, got {args.level}", file=sys.stderr)
        sys.exit(1)

    # Determine fragment file path
    if args.fragment_file:
        fragment_file = args.fragment_file
        cleanup_fragments = False
    elif args.keep_fragments:
        input_path = Path(args.input)
        fragment_file = str(input_path.with_suffix('.fragments'))
        cleanup_fragments = False
    else:
        # Use temporary file
        fd, fragment_file = tempfile.mkstemp(suffix='.fragments')
        os.close(fd)
        cleanup_fragments = True

    try:
        # Get initial GPU stats
        gpu_stats_before = get_gpu_stats()

        total_start = time.time()

        # Step 1: Convert PLY to fragments
        conversion_start = time.time()
        if not run_conversion(args.input, fragment_file, args.level, args.no_dedup):
            sys.exit(1)
        conversion_time = time.time() - conversion_start

        # Step 2: Compress with octree
        compression_start = time.time()
        if not run_compression(fragment_file, args.level, args.output):
            sys.exit(1)
        compression_time = time.time() - compression_start

        total_time = time.time() - total_start

        # Get final GPU stats
        gpu_stats_after = get_gpu_stats()

        print("\n" + "=" * 60)
        print("Compression complete!")
        print("=" * 60)

        # Timing summary
        print(f"\nTiming:")
        print(f"  PLY to fragments:  {conversion_time*1000:.2f} ms")
        print(f"  Octree building:   {compression_time*1000:.2f} ms")
        print(f"  Total:             {total_time*1000:.2f} ms")

        if args.output:
            octree_size = Path(args.output).stat().st_size
            fragment_size = Path(fragment_file).stat().st_size
            print(f"\nCompression:")
            print(f"  Fragment file: {fragment_size / (1024*1024):.2f} MB")
            print(f"  Octree file:   {octree_size / (1024*1024):.2f} MB")
            print(f"  Ratio:         {fragment_size / octree_size:.2f}x")

        # GPU stats summary
        if gpu_stats_before and gpu_stats_after:
            print(f"\nGPU Status:")
            for i, (before, after) in enumerate(zip(gpu_stats_before, gpu_stats_after)):
                print(f"  GPU {before['index']} ({before['name']}):")
                print(f"    Memory: {before['memory_used']} MB -> {after['memory_used']} MB "
                      f"(+{after['memory_used'] - before['memory_used']} MB) / {after['memory_total']} MB")
                print(f"    Utilization: {before['utilization']}% -> {after['utilization']}%")

    finally:
        # Cleanup temporary fragment file
        if cleanup_fragments and Path(fragment_file).exists():
            Path(fragment_file).unlink()


if __name__ == '__main__':
    main()
