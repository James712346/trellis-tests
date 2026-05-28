#!/usr/bin/env python3
"""
Generate probability tables for Trellis sparse HMM
Trellis handles sparsity through set constraints, so we only provide
the tables it needs - not full transition matrices!
"""

import numpy as np

# Configuration matching Trellis model
GRID_WIDTH = 48
GRID_HEIGHT = 27
NUM_BUCKETS = 8
NUM_TEXTURES = 4

NUM_POSITIONS = GRID_WIDTH * GRID_HEIGHT
TOTAL_STATES = NUM_POSITIONS * NUM_BUCKETS * NUM_TEXTURES

print("=" * 70)
print("TRELLIS SPARSE HMM TABLE GENERATOR")
print("=" * 70)
print(f"Configuration:")
print(f"  Grid: {GRID_WIDTH}×{GRID_HEIGHT} = {NUM_POSITIONS} positions")
print(f"  Intensity buckets: {NUM_BUCKETS}")
print(f"  Texture types: {NUM_TEXTURES}")
print(f"  Total states: {TOTAL_STATES:,}")
print("=" * 70)


def generate_initial_prob():
    """Generate initial state probabilities"""
    print("\nGenerating initial probabilities...")
    
    # Uniform distribution over all states
    initial = np.ones((NUM_POSITIONS, NUM_BUCKETS, NUM_TEXTURES))
    initial = initial / initial.sum()
    
    print(f"  Shape: {initial.shape}")
    print(f"  Memory: {initial.nbytes / 1024:.2f} KB")
    
    return initial


def generate_emission_prob():
    """
    Generate emission probabilities: P(intensity | intensity_bucket)
    Each bucket has a Gaussian distribution
    """
    print("\nGenerating emission probabilities...")
    
    emission = np.zeros((256, NUM_BUCKETS))
    
    for bucket in range(NUM_BUCKETS):
        # Center of bucket
        bucket_center = bucket * (256 // NUM_BUCKETS) + (256 // NUM_BUCKETS // 2)
        
        # Gaussian around bucket center
        for intensity in range(256):
            emission[intensity, bucket] = np.exp(-0.05 * (intensity - bucket_center)**2)
    
    # Normalize each bucket
    for bucket in range(NUM_BUCKETS):
        emission[:, bucket] /= emission[:, bucket].sum()
    
    print(f"  Shape: {emission.shape}")
    print(f"  Memory: {emission.nbytes / 1024:.2f} KB")
    
    return emission


def generate_spatial_trans():
    """
    Generate spatial transition probabilities
    Only for local transitions (adjacent positions)
    """
    print("\nGenerating spatial transition probabilities...")
    
    # Small table for spatial transitions
    # Most transitions have zero probability (sparse!)
    spatial = np.zeros((NUM_POSITIONS, NUM_POSITIONS))
    
    for p1 in range(NUM_POSITIONS):
        y1 = p1 // GRID_WIDTH
        x1 = p1 % GRID_WIDTH
        
        # Stay in same position
        spatial[p1, p1] = 0.5
        
        # Move to adjacent positions
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            y2 = y1 + dy
            x2 = x1 + dx
            
            if 0 <= y2 < GRID_HEIGHT and 0 <= x2 < GRID_WIDTH:
                p2 = y2 * GRID_WIDTH + x2
                spatial[p1, p2] = 0.125  # Distribute remaining 0.5 to neighbors
    
    # Normalize
    row_sums = spatial.sum(axis=1, keepdims=True)
    spatial = spatial / row_sums
    
    # Count non-zeros (to show sparsity)
    nnz = np.count_nonzero(spatial)
    sparsity = (1 - nnz / spatial.size) * 100
    
    print(f"  Shape: {spatial.shape}")
    print(f"  Non-zeros: {nnz:,} / {spatial.size:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Memory: {spatial.nbytes / (1024**2):.2f} MB")
    
    return spatial


def generate_intensity_trans():
    """Generate intensity bucket transition probabilities"""
    print("\nGenerating intensity transition probabilities...")
    
    intensity_trans = np.zeros((NUM_BUCKETS, NUM_BUCKETS))
    
    for b1 in range(NUM_BUCKETS):
        # Stay in same bucket (most likely)
        intensity_trans[b1, b1] = 0.6
        
        # Move to adjacent buckets
        if b1 > 0:
            intensity_trans[b1, b1 - 1] = 0.2
        if b1 < NUM_BUCKETS - 1:
            intensity_trans[b1, b1 + 1] = 0.2
    
    # Normalize
    row_sums = intensity_trans.sum(axis=1, keepdims=True)
    intensity_trans = intensity_trans / row_sums
    
    print(f"  Shape: {intensity_trans.shape}")
    print(f"  Memory: {intensity_trans.nbytes} bytes")
    
    return intensity_trans


def generate_texture_trans():
    """Generate texture type transition probabilities"""
    print("\nGenerating texture transition probabilities...")
    
    texture_trans = np.zeros((NUM_TEXTURES, NUM_TEXTURES))
    
    for t1 in range(NUM_TEXTURES):
        # Stay in same texture (most likely)
        texture_trans[t1, t1] = 0.7
        
        # Move to adjacent textures
        if t1 > 0:
            texture_trans[t1, t1 - 1] = 0.15
        if t1 < NUM_TEXTURES - 1:
            texture_trans[t1, t1 + 1] = 0.15
    
    # Normalize
    row_sums = texture_trans.sum(axis=1, keepdims=True)
    texture_trans = texture_trans / row_sums
    
    print(f"  Shape: {texture_trans.shape}")
    print(f"  Memory: {texture_trans.nbytes} bytes")
    
    return texture_trans


def main():
    print("\n🔥 Generating Tables for Trellis Sparse HMM 🔥\n")
    
    # Generate all tables
    initial_prob = generate_initial_prob()
    emission_prob = generate_emission_prob()
    spatial_trans = generate_spatial_trans()
    intensity_trans = generate_intensity_trans()
    texture_trans = generate_texture_trans()
    
    # Save tables
    print("\n" + "=" * 70)
    print("SAVING TABLES")
    print("=" * 70)
    
    np.save('initialProb.npy', initial_prob)
    print(f"✓ Saved initialProb.npy")
    
    np.save('emissionProb.npy', emission_prob)
    print(f"✓ Saved emissionProb.npy")
    
    np.save('spatialTrans.npy', spatial_trans)
    print(f"✓ Saved spatialTrans.npy")
    
    np.save('intensityTrans.npy', intensity_trans)
    print(f"✓ Saved intensityTrans.npy")
    
    np.save('textureTrans.npy', texture_trans)
    print(f"✓ Saved textureTrans.npy")
    
    # Calculate total memory
    total_memory = (initial_prob.nbytes + emission_prob.nbytes + 
                   spatial_trans.nbytes + intensity_trans.nbytes + 
                   texture_trans.nbytes)
    
    # Compare to dense transition matrix
    dense_size = TOTAL_STATES ** 2 * 8  # 8 bytes per float64
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total table memory: {total_memory / (1024**2):.2f} MB")
    print(f"Dense transition matrix would be: {dense_size / (1024**3):.2f} GB")
    print(f"Memory savings: {(1 - total_memory / dense_size) * 100:.2f}%")
    print("\n✅ Trellis handles the sparsity through set constraints!")
    print("   No need to generate massive sparse matrices.")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Compile Trellis model: trellis image_hmm.trellis")
    print("  2. Run: python trellis_runner.py")


if __name__ == "__main__":
    main()
