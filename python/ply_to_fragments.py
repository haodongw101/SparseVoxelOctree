#!/usr/bin/env python3
"""
Convert PLY file with voxelized points to binary voxel fragment format.

Voxel Fragment Format (8 bytes per fragment):
    uint32_t[0]: X (bits 0-11) | Y (bits 12-23) | Z_low (bits 24-31)
    uint32_t[1]: RGB_24bit (bits 0-23) | Z_high (bits 28-31)
"""

import argparse
import struct
import sys
from pathlib import Path
import numpy as np


def read_ply(filename):
    """
    Read a PLY file and extract voxel positions and colors.

    Returns:
        positions: Nx3 array of (x, y, z) positions
        colors: Nx3 array of (r, g, b) colors in range [0, 255]
    """
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("Error: plyfile library not found. Install with: pip install plyfile", file=sys.stderr)
        sys.exit(1)

    print(f"Reading PLY file: {filename}")
    plydata = PlyData.read(filename)

    vertex = plydata['vertex']
    positions = np.column_stack([vertex['x'], vertex['y'], vertex['z']])

    # Check if colors are present
    has_colors = all(k in vertex for k in ['red', 'green', 'blue'])

    if has_colors:
        colors = np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
        # Normalize colors to 0-255 range if needed
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)
    else:
        # Default to white if no colors
        print("Warning: No color information found, using white (255, 255, 255)")
        colors = np.full((len(positions), 3), 255, dtype=np.uint8)

    print(f"Loaded {len(positions)} points")
    return positions, colors


def voxelize_points(positions, colors, voxel_resolution):
    """
    Convert point cloud positions to voxel grid coordinates.

    Args:
        positions: Nx3 array of point positions
        colors: Nx3 array of colors
        voxel_resolution: Target voxel grid resolution (e.g., 1024 for 10-bit)

    Returns:
        voxel_coords: Nx3 array of integer voxel coordinates [0, voxel_resolution)
        voxel_colors: Nx3 array of colors
    """
    # Normalize positions to [0, 1] range
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)

    print(f"Point cloud bounds: min={min_pos}, max={max_pos}")

    # Normalize to [0, 1]
    normalized = (positions - min_pos) / (max_pos - min_pos + 1e-10)

    # Scale to voxel grid, leaving a small margin to avoid overflow
    voxel_coords = (normalized * (voxel_resolution - 1)).astype(np.uint32)

    # Clamp to valid range
    voxel_coords = np.clip(voxel_coords, 0, voxel_resolution - 1)

    print(f"Voxel coordinate range: [{voxel_coords.min()}, {voxel_coords.max()}]")

    return voxel_coords, colors


def remove_duplicates(voxel_coords, colors):
    """
    Remove duplicate voxels, averaging colors for duplicates.

    Returns:
        unique_coords: Mx3 array of unique voxel coordinates
        averaged_colors: Mx3 array of averaged colors
    """
    # Combine coordinates into a single value for duplicate detection
    # Using lexicographic ordering
    voxel_keys = (voxel_coords[:, 0].astype(np.int64) << 24) | \
                 (voxel_coords[:, 1].astype(np.int64) << 12) | \
                 voxel_coords[:, 2].astype(np.int64)

    unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)

    # Average colors for duplicate voxels
    unique_colors = np.zeros((len(unique_keys), 3), dtype=np.float32)
    counts = np.zeros(len(unique_keys), dtype=np.uint32)

    for i, idx in enumerate(inverse_indices):
        unique_colors[idx] += colors[i].astype(np.float32)
        counts[idx] += 1

    unique_colors = (unique_colors / counts[:, np.newaxis]).astype(np.uint8)

    # Reconstruct coordinates
    unique_coords = np.column_stack([
        (unique_keys >> 24) & 0xFFF,
        (unique_keys >> 12) & 0xFFF,
        unique_keys & 0xFFF
    ]).astype(np.uint32)

    print(f"Removed {len(voxel_coords) - len(unique_coords)} duplicate voxels")
    print(f"Unique voxels: {len(unique_coords)}")

    return unique_coords, unique_colors


def pack_fragment(x, y, z, r, g, b):
    """
    Pack voxel position and color into two uint32 values.

    Format:
        uint32_0: X (bits 0-11) | Y (bits 12-23) | Z_low (bits 24-31)
        uint32_1: RGB (bits 0-23) | Z_high (bits 28-31)
    """
    # Pack RGB into 24 bits
    rgb_packed = (b << 16) | (g << 8) | r

    # Pack coordinates
    uint32_0 = (x & 0xFFF) | ((y & 0xFFF) << 12) | ((z & 0xFF) << 24)
    uint32_1 = (rgb_packed & 0xFFFFFF) | ((z >> 8) << 28)

    return uint32_0, uint32_1


def write_fragments(output_file, voxel_coords, colors):
    """
    Write voxel fragments to binary file.

    Args:
        output_file: Path to output file
        voxel_coords: Nx3 array of voxel coordinates
        colors: Nx3 array of colors
    """
    print(f"Writing {len(voxel_coords)} fragments to {output_file}")

    with open(output_file, 'wb') as f:
        for i in range(len(voxel_coords)):
            x, y, z = voxel_coords[i]
            r, g, b = colors[i]

            uint32_0, uint32_1 = pack_fragment(x, y, z, r, g, b)

            # Write as little-endian uint32
            f.write(struct.pack('<I', uint32_0))
            f.write(struct.pack('<I', uint32_1))

    file_size = Path(output_file).stat().st_size
    print(f"Wrote {file_size / (1024*1024):.2f} MB to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert PLY file to voxel fragment format')
    parser.add_argument('-i', '--input', required=True, help='Input PLY file')
    parser.add_argument('-o', '--output', required=True, help='Output fragment file')
    parser.add_argument('-lvl', '--level', type=int, required=True,
                        help='Octree level (determines voxel resolution = 2^level)')
    parser.add_argument('--no-dedup', action='store_true',
                        help='Skip duplicate removal (faster but may create larger files)')

    args = parser.parse_args()

    if args.level < 1 or args.level > 12:
        print(f"Error: Octree level must be between 1 and 12, got {args.level}", file=sys.stderr)
        sys.exit(1)

    voxel_resolution = 1 << args.level
    print(f"Octree level: {args.level} (voxel resolution: {voxel_resolution}^3)")

    # Read PLY file
    positions, colors = read_ply(args.input)

    # Voxelize
    voxel_coords, voxel_colors = voxelize_points(positions, colors, voxel_resolution)

    # Remove duplicates
    if not args.no_dedup:
        voxel_coords, voxel_colors = remove_duplicates(voxel_coords, voxel_colors)

    # Write to binary file
    write_fragments(args.output, voxel_coords, voxel_colors)

    print("Conversion complete!")


if __name__ == '__main__':
    main()
