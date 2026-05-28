#!/usr/bin/env python3
"""
Converts HMM probability tables from numpy (.npy / .npz) exports into
MATLAB-compatible .mat files (v7.3 / HDF5), matching the format expected
by generate_tables.m.

Usage:
    python convert_numpy_tables.py <model_name> [--src-root PATH] [--dst-root PATH]

    model_name   e.g. SMALL, MEDIUM, LARGE …
    --src-root   parent of  <model>/tables/  with the .npy files
                 default: ../image-sparse/models
    --dst-root   parent of  <model>/tables/  for the .mat output
                 default: models

Example:
    python convert_numpy_tables.py SMALL
    python convert_numpy_tables.py LARGE --src-root /data/image-sparse/models --dst-root ./models
"""

import argparse
import json
import os
import sys
import numpy as np
import scipy.io
import scipy.sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npy(path: str) -> np.ndarray:
    """Load a .npy file, converting to float64 (MATLAB default)."""
    arr = np.load(path, allow_pickle=False)
    return arr.astype(np.float64)


def load_sparse(path: str) -> scipy.sparse.csc_matrix:
    """
    Load spatialWeight.  The numpy export may be:
      - a .npz file saved with scipy.sparse.save_npz  → load directly
      - a dense .npy file                              → convert to sparse
    """
    if path.endswith('.npz'):
        return scipy.sparse.load_npz(path).tocsc().astype(np.float64)
    arr = load_npy(path)
    return scipy.sparse.csc_matrix(arr)


def save_mat(out_path: str, data) -> None:
    """Save a single variable named 'data' to a .mat file (v7.3)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scipy.io.savemat(out_path, {'data': data}, format='5', do_compression=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f'  saved {out_path}  ({size_kb:.2f} KB)')


def matlab_column_major(arr: np.ndarray) -> np.ndarray:
    """
    MATLAB stores arrays in column-major (Fortran) order.
    scipy.io.savemat handles this automatically, but we make sure the
    array is C-contiguous first so the conversion is unambiguous.
    """
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(model_name: str, src_root: str, dst_root: str) -> None:
    model_lower = model_name.lower()
    src_dir = os.path.join(src_root, model_lower, 'tables')
    dst_dir = os.path.join(dst_root, model_lower, 'tables')

    if not os.path.isdir(src_dir):
        sys.exit(f'ERROR: source directory not found: {src_dir}')

    os.makedirs(dst_dir, exist_ok=True)

    print('=' * 55)
    print('NUMPY → MAT TABLE CONVERTER')
    print('=' * 55)
    print(f'Model   : {model_name}')
    print(f'Source  : {src_dir}')
    print(f'Dest    : {dst_dir}')
    print('=' * 55)

    # ------------------------------------------------------------------
    # 1. Simple dense tables
    # ------------------------------------------------------------------
    dense_tables = [
        'initialProb',
        'emissionProb',
        'intensityWeight',
        'textureWeight',
        'positionBias',
        'bucketBias',
        'textureBias',
    ]

    for name in dense_tables:
        npy_path = os.path.join(src_dir, f'{name}.npy')
        if not os.path.exists(npy_path):
            print(f'  WARNING: {npy_path} not found – skipping')
            continue

        arr = load_npy(npy_path)

        # MATLAB uses 1-based indexing but that's irrelevant for raw arrays.
        # initialProb shape from Python is (np, b, t); MATLAB expects the
        # same logical layout – scipy.io.savemat will transpose axes so that
        # MATLAB sees (np, b, t) when indexing normally.
        arr = matlab_column_major(arr)
        save_mat(os.path.join(dst_dir, f'{name}.mat'), arr)

    # ------------------------------------------------------------------
    # 2. Spatial weight (sparse)
    # ------------------------------------------------------------------
    # Try .npz first (scipy sparse), fall back to dense .npy
    spw_npz = os.path.join(src_dir, 'spatialWeight.npz')
    spw_npy = os.path.join(src_dir, 'spatialWeight.npy')

    if os.path.exists(spw_npz):
        spw = load_sparse(spw_npz)
        print(f'  loading spatialWeight from .npz (sparse)')
    elif os.path.exists(spw_npy):
        spw = load_sparse(spw_npy)
        print(f'  loading spatialWeight from .npy (dense → sparse)')
    else:
        print(f'  WARNING: spatialWeight not found – skipping')
        spw = None

    if spw is not None:
        nnz = spw.nnz
        total = spw.shape[0] * spw.shape[1]
        sparsity = (1 - nnz / total) * 100
        print(f'  shape: {spw.shape}  nnz: {nnz}  sparsity: {sparsity:.4f}%')
        save_mat(os.path.join(dst_dir, 'spatialWeight.mat'), spw)

    # ------------------------------------------------------------------
    # 3. Metadata
    # ------------------------------------------------------------------
    meta_json = os.path.join(src_dir, 'metadata.json')
    if os.path.exists(meta_json):
        with open(meta_json) as f:
            meta = json.load(f)

        # Build a MATLAB-friendly struct via nested dict
        matlab_meta = {
            'config_name': model_name,
            'config': {k: v for k, v in meta.items() if k != 'config_name'},
        }
        scipy.io.savemat(
            os.path.join(dst_dir, 'metadata.mat'),
            {'metadata': matlab_meta},
            format='5',
        )
        print(f'  saved {os.path.join(dst_dir, "metadata.mat")}')
    else:
        print(f'  WARNING: metadata.json not found – skipping metadata.mat')

    print('\nDone.\n')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert numpy HMM tables to .mat')
    parser.add_argument('model_name', help='Config name, e.g. SMALL, MEDIUM')
    parser.add_argument('--src-root', default='../image-sparse/models',
                        help='Parent directory of <model>/tables/ (numpy source)')
    parser.add_argument('--dst-root', default='models',
                        help='Parent directory of <model>/tables/ (mat output)')
    args = parser.parse_args()

    convert(args.model_name.upper(), args.src_root, args.dst_root)
