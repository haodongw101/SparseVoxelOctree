# Point Cloud Compression Pipeline

This directory contains Python tools for compressing voxelized point clouds using Sparse Voxel Octrees.

## Overview

The pipeline consists of:
1. **PLY to Fragments Converter** (`ply_to_fragments.py`) - Converts PLY files to binary voxel fragment format
2. **Octree Compressor** (`octree_compressor`) - C++ tool that builds sparse voxel octrees from fragments
3. **End-to-End Wrapper** (`compress_pointcloud.py`) - Orchestrates the full pipeline

## Installation

### Prerequisites

- Python 3.6+
- CMake 3.15+
- Vulkan SDK
- C++20 compiler

### Python Dependencies

```bash
pip install plyfile numpy
```

### Build the C++ Compressor

```bash
cd ..  # Go to repository root
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make octree_compressor
```

## Usage

### Quick Start - End-to-End Compression

```bash
# Optional: generate a ply for test
python example_create_test_ply.py -s 50 -o test_cube.ply

# Compress a point cloud at octree level 10 (1024^3 voxel resolution)
python compress_pointcloud.py -i input.ply -lvl 10

# Compress and save the octree
python compress_pointcloud.py -i input.ply -lvl 10 -o compressed.octree

# Keep intermediate fragment file for inspection
python compress_pointcloud.py -i input.ply -lvl 10 --keep-fragments
```

### Step-by-Step Usage

#### Step 1: Convert PLY to Fragments

```bash
python ply_to_fragments.py -i input.ply -o output.fragments -lvl 10
```

Options:
- `-i, --input`: Input PLY file
- `-o, --output`: Output fragment file (binary)
- `-lvl, --level`: Octree level (1-12), determines voxel resolution = 2^level
- `--no-dedup`: Skip duplicate voxel removal (faster but larger)

#### Step 2: Compress with Octree

```bash
../build/octree_compressor -i output.fragments -lvl 10 -o compressed.octree
```

Options:
- `-i`: Input fragment file
- `-lvl`: Octree level (must match the level used in Step 1)
- `-o`: Output octree file (optional)

## Voxel Fragment Format

The intermediate fragment format is a binary file with 8 bytes per voxel:

```
uint32_t[0]: X (bits 0-11) | Y (bits 12-23) | Z_low (bits 24-31)
uint32_t[1]: RGB (bits 0-23) | Z_high (bits 28-31)
```

- Supports up to 4096^3 voxel resolution (12 bits per coordinate)
- 24-bit RGB color per voxel
- Little-endian encoding

## Octree Levels

| Level | Resolution | Max Voxels | Recommended For |
|-------|------------|------------|-----------------|
| 8     | 256^3      | 16M        | Small objects   |
| 10    | 1024^3     | 1B         | Medium scenes   |
| 11    | 2048^3     | 8B         | Large scenes    |
| 12    | 4096^3     | 64B        | Very large      |

## Input PLY Requirements

- Must contain vertex positions (`x`, `y`, `z`)
- Optional: vertex colors (`red`, `green`, `blue`)
  - If colors are in range [0, 1], they will be scaled to [0, 255]
  - If no colors are present, white (255, 255, 255) is used

## Output

The compressor outputs:
- Build time in milliseconds
- Number of octree nodes
- Compressed size in MB
- Compression ratio (voxels → nodes)

Example output:
```
[12:34:56.789] [info] Octree built successfully in 156 ms
[12:34:56.789] [info] Octree size: 125432 nodes (4.01 MB)
[12:34:56.789] [info] Compression ratio: 8.25x (from 1034567 voxels to 125432 nodes)
```

## Performance Tips

1. **Choose appropriate octree level**: Higher levels = more detail but slower build times
2. **Remove duplicates**: Use duplicate removal (default) to reduce redundant voxels
3. **Pre-filter point clouds**: Remove outliers and noise before conversion
4. **GPU selection**: The compressor uses the first available Vulkan compute device

## Troubleshooting

### "Failed to load vulkan"
- Ensure Vulkan SDK is installed and `VK_SDK_PATH` is set
- Check that your GPU supports Vulkan

### "Failed to open fragment file"
- Verify the fragment file exists and has correct permissions
- Check that octree level matches between conversion and compression steps

### "Invalid fragment file size"
- File may be corrupted
- Ensure file is a multiple of 8 bytes (2 × uint32 per voxel)

## Examples

### Example 1: Basic Compression
```bash
python compress_pointcloud.py -i pointcloud.ply -lvl 10
```

### Example 2: Save Everything
```bash
python compress_pointcloud.py \
    -i pointcloud.ply \
    -lvl 10 \
    --keep-fragments \
    -o compressed.octree
```

### Example 3: Manual Two-Step Process
```bash
# Step 1: Convert
python ply_to_fragments.py -i input.ply -o temp.fragments -lvl 10

# Step 2: Compress
../build/octree_compressor -i temp.fragments -lvl 10 -o output.octree
```

## Technical Details

The sparse voxel octree compression works by:
1. Converting points to a uniform voxel grid
2. Building a hierarchical octree structure
3. Storing only occupied nodes (sparse representation)
4. Averaging colors for overlapping voxels

Typical compression ratios: 5x-20x depending on point cloud sparsity.
