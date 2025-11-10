#!/usr/bin/env python3
"""
Create a simple test PLY file with a cube of voxelized points.
This can be used to test the compression pipeline.
"""

import numpy as np
import sys

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    print("Error: plyfile library not found. Install with: pip install plyfile", file=sys.stderr)
    sys.exit(1)


def create_cube_pointcloud(size=100, output_file="test_cube.ply"):
    """
    Create a cube-shaped point cloud.

    Args:
        size: Number of points along each edge
        output_file: Output PLY filename
    """
    print(f"Creating {size}x{size}x{size} cube point cloud...")

    # Generate cube grid
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    z = np.linspace(0, 1, size)

    # Create hollow cube (only surfaces)
    points = []
    colors = []

    # Bottom and top faces (z=0 and z=1)
    for i, zi in enumerate([z[0], z[-1]]):
        for xi in x:
            for yi in y:
                points.append([xi, yi, zi])
                # Color based on height
                colors.append([int(255 * (1 - i)), int(128), int(255 * i)])

    # Front and back faces (y=0 and y=1)
    for i, yi in enumerate([y[0], y[-1]]):
        for xi in x:
            for zi in z[1:-1]:  # Exclude edges already covered
                points.append([xi, yi, zi])
                colors.append([int(255 * yi), int(255 * (1-yi)), 128])

    # Left and right faces (x=0 and x=1)
    for i, xi in enumerate([x[0], x[-1]]):
        for yi in y[1:-1]:  # Exclude edges already covered
            for zi in z[1:-1]:
                points.append([xi, yi, zi])
                colors.append([int(255 * xi), 128, int(255 * (1-xi))])

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    print(f"Generated {len(points)} points")

    # Create structured array for PLY
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    vertex_data = np.zeros(len(points), dtype=vertex_dtype)
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]

    # Create PLY element and save
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(output_file)

    print(f"Saved to {output_file}")
    return output_file


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create test PLY file with cube point cloud')
    parser.add_argument('-s', '--size', type=int, default=50,
                        help='Cube size (points per edge, default: 50)')
    parser.add_argument('-o', '--output', default='test_cube.ply',
                        help='Output PLY file (default: test_cube.ply)')

    args = parser.parse_args()

    create_cube_pointcloud(args.size, args.output)
    print("\nTest with:")
    print(f"  python compress_pointcloud.py -i {args.output} -lvl 8")
