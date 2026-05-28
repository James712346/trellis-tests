#!/usr/bin/env python3
"""
Comprehensive Trellis Table Generator
Generates probability tables for any Trellis configuration
"""

import numpy as np
import sys
import json
import os

# Load configurations
try:
    with open('trellis_generation_results.json', 'r') as f:
        RESULTS = json.load(f)
        CONFIGS = {name: result['config'] for name, result in RESULTS.items()}
except:
    # Fallback configurations
    CONFIGS = {
        'SMALL': {'grid_width': 32, 'grid_height': 18, 'num_buckets': 8, 'num_textures': 4},
        'MEDIUM': {'grid_width': 48, 'grid_height': 27, 'num_buckets': 8, 'num_textures': 4},
        'LARGE': {'grid_width': 64, 'grid_height': 36, 'num_buckets': 8, 'num_textures': 4},
    }


class TableGenerator:
    """Generate probability tables for Trellis HMM"""
    
    def __init__(self, config):
        """Initialize with configuration"""
        self.grid_width = config['grid_width']
        self.grid_height = config['grid_height']
        self.num_buckets = config['num_buckets']
        self.num_textures = config['num_textures']
        
        self.num_positions = self.grid_width * self.grid_height
        self.total_states = self.num_positions * self.num_buckets * self.num_textures
        
        print("=" * 70)
        print("TRELLIS TABLE GENERATOR")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Grid: {self.grid_width}×{self.grid_height} = {self.num_positions}")
        print(f"  Buckets: {self.num_buckets}")
        print(f"  Textures: {self.num_textures}")
        print(f"  Total states: {self.total_states:,}")
        print("=" * 70)
    
    def generate_initial_prob(self):
        """Generate initial state probabilities"""
        print("\n[1/7] Generating initialProb...")
        
        # Create probability distribution favoring center positions
        initial = np.zeros((self.num_positions, self.num_buckets, self.num_textures))
        
        for pos in range(self.num_positions):
            y = pos // self.grid_width
            x = pos % self.grid_width
            
            # Distance from center
            center_y = self.grid_height / 2
            center_x = self.grid_width / 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Gaussian centered on image center
            position_prob = np.exp(-0.5 * (dist / (max_dist / 3))**2)
            
            for bucket in range(self.num_buckets):
                # Favor mid-range intensity buckets initially
                bucket_center = self.num_buckets / 2
                bucket_prob = np.exp(-0.3 * (bucket - bucket_center)**2)
                
                for texture in range(self.num_textures):
                    # Uniform texture distribution
                    initial[pos, bucket, texture] = position_prob * bucket_prob
        
        # Normalize
        initial = initial / initial.sum()
        
        print(f"  Shape: {initial.shape}")
        print(f"  Memory: {initial.nbytes / 1024:.2f} KB")
        print(f"  Min: {initial.min():.6e}, Max: {initial.max():.6e}")
        
        return initial
    
    def generate_emission_prob(self):
        """Generate emission probabilities P(intensity | bucket)"""
        print("\n[2/7] Generating emissionProb...")
        
        emission = np.zeros((256, self.num_buckets))
        
        for bucket in range(self.num_buckets):
            # Each bucket covers a range of intensities
            bucket_size = 256 // self.num_buckets
            bucket_center = bucket * bucket_size + bucket_size // 2
            
            # Gaussian distribution around bucket center
            sigma = bucket_size / 2
            
            for intensity in range(256):
                emission[intensity, bucket] = np.exp(-0.5 * ((intensity - bucket_center) / sigma)**2)
        
        # Normalize each bucket column
        for bucket in range(self.num_buckets):
            emission[:, bucket] /= emission[:, bucket].sum()
        
        print(f"  Shape: {emission.shape}")
        print(f"  Memory: {emission.nbytes / 1024:.2f} KB")
        print(f"  Non-zeros: {np.count_nonzero(emission):,} / {emission.size:,}")
        
        return emission
    
    def generate_spatial_weight(self):
        """Generate spatial transition weights"""
        print("\n[3/7] Generating spatialWeight...")
        print(f"  This is the LARGEST table - {self.num_positions}×{self.num_positions}")
        
        spatial = np.zeros((self.num_positions, self.num_positions))
        
        for p1 in range(self.num_positions):
            if p1 % 100 == 0:
                print(f"    Progress: {p1:,} / {self.num_positions:,} ({p1/self.num_positions*100:.1f}%)")
            
            y1 = p1 // self.grid_width
            x1 = p1 % self.grid_width
            
            for p2 in range(self.num_positions):
                y2 = p2 // self.grid_width
                x2 = p2 % self.grid_width
                
                # Manhattan distance
                dist = abs(x2 - x1) + abs(y2 - y1)
                
                # Only allow local transitions (within distance 2)
                if dist == 0:
                    # Stay in place (most likely)
                    spatial[p1, p2] = 0.5
                elif dist == 1:
                    # Adjacent (4-connected)
                    spatial[p1, p2] = 0.3
                elif dist == 2:
                    # Diagonal or 2 steps away
                    # Check if it's truly diagonal
                    if abs(x2 - x1) == 1 and abs(y2 - y1) == 1:
                        spatial[p1, p2] = 0.15
                    elif abs(x2 - x1) == 2 or abs(y2 - y1) == 2:
                        spatial[p1, p2] = 0.1
                # Beyond distance 2: probability = 0 (sparse!)
        
        # Normalize rows
        print(f"    Normalizing rows...")
        row_sums = spatial.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        spatial = spatial / row_sums
        
        # Calculate sparsity
        nnz = np.count_nonzero(spatial)
        sparsity = (1 - nnz / spatial.size) * 100
        
        print(f"  Shape: {spatial.shape}")
        print(f"  Non-zeros: {nnz:,} / {spatial.size:,}")
        print(f"  Sparsity: {sparsity:.4f}%")
        print(f"  Memory: {spatial.nbytes / (1024**2):.2f} MB")
        print(f"  Avg non-zeros per row: {nnz / self.num_positions:.1f}")
        
        return spatial
    
    def generate_intensity_weight(self):
        """Generate intensity bucket transition weights"""
        print("\n[4/7] Generating intensityWeight...")
        
        intensity = np.zeros((self.num_buckets, self.num_buckets))
        
        for b1 in range(self.num_buckets):
            for b2 in range(self.num_buckets):
                diff = abs(b2 - b1)
                
                if diff == 0:
                    # Stay in same bucket
                    intensity[b1, b2] = 0.6
                elif diff == 1:
                    # Adjacent bucket
                    intensity[b1, b2] = 0.25
                elif diff == 2:
                    # 2 buckets away
                    intensity[b1, b2] = 0.1
                # Beyond 2: probability = 0 (sparse)
        
        # Normalize rows
        row_sums = intensity.sum(axis=1, keepdims=True)
        intensity = intensity / row_sums
        
        nnz = np.count_nonzero(intensity)
        sparsity = (1 - nnz / intensity.size) * 100
        
        print(f"  Shape: {intensity.shape}")
        print(f"  Non-zeros: {nnz} / {intensity.size}")
        print(f"  Sparsity: {sparsity:.2f}%")
        print(f"  Memory: {intensity.nbytes} bytes")
        
        return intensity
    
    def generate_texture_weight(self):
        """Generate texture type transition weights"""
        print("\n[5/7] Generating textureWeight...")
        
        texture = np.zeros((self.num_textures, self.num_textures))
        
        for t1 in range(self.num_textures):
            for t2 in range(self.num_textures):
                diff = abs(t2 - t1)
                
                if diff == 0:
                    # Stay in same texture
                    texture[t1, t2] = 0.7
                elif diff == 1:
                    # Adjacent texture
                    texture[t1, t2] = 0.2
                # Beyond 1: probability = 0 (sparse)
        
        # Normalize rows
        row_sums = texture.sum(axis=1, keepdims=True)
        texture = texture / row_sums
        
        nnz = np.count_nonzero(texture)
        sparsity = (1 - nnz / texture.size) * 100
        
        print(f"  Shape: {texture.shape}")
        print(f"  Non-zeros: {nnz} / {texture.size}")
        print(f"  Sparsity: {sparsity:.2f}%")
        print(f"  Memory: {texture.nbytes} bytes")
        
        return texture
    
    def generate_position_bias(self):
        """Generate position-specific bias"""
        print("\n[6/7] Generating positionBias...")
        
        bias = np.ones(self.num_positions)
        
        # Bias toward edges (simulating image borders)
        for pos in range(self.num_positions):
            y = pos // self.grid_width
            x = pos % self.grid_width
            
            # Distance to nearest edge
            dist_to_edge = min(x, self.grid_width - 1 - x, y, self.grid_height - 1 - y)
            max_dist = min(self.grid_width, self.grid_height) // 2
            
            # Edges have slightly higher bias
            bias[pos] = 0.8 + 0.4 * (1 - dist_to_edge / max_dist)
        
        # Normalize
        bias = bias / bias.sum()
        
        print(f"  Shape: {bias.shape}")
        print(f"  Memory: {bias.nbytes} bytes")
        print(f"  Range: [{bias.min():.6f}, {bias.max():.6f}]")
        
        return bias
    
    def generate_bucket_bias(self):
        """Generate intensity bucket bias"""
        print("\n[7/7] Generating bucketBias...")
        
        bias = np.ones(self.num_buckets)
        
        # Favor mid-range intensities
        for bucket in range(self.num_buckets):
            bucket_center = self.num_buckets / 2
            bias[bucket] = np.exp(-0.2 * (bucket - bucket_center)**2)
        
        # Normalize
        bias = bias / bias.sum()
        
        print(f"  Shape: {bias.shape}")
        print(f"  Memory: {bias.nbytes} bytes")
        print(f"  Range: [{bias.min():.6f}, {bias.max():.6f}]")
        
        return bias
    
    def generate_texture_bias(self):
        """Generate texture type bias"""
        print("\n[Bonus] Generating textureBias...")
        
        # Uniform bias for textures
        bias = np.ones(self.num_textures) / self.num_textures
        
        print(f"  Shape: {bias.shape}")
        print(f"  Memory: {bias.nbytes} bytes")
        
        return bias
    
    def generate_all(self, output_dir='.'):
        """Generate all tables and save to files"""
        print("\n" + "=" * 70)
        print("GENERATING ALL TABLES")
        print("=" * 70)
        
        tables = {}
        
        # Generate each table
        tables['initialProb'] = self.generate_initial_prob()
        tables['emissionProb'] = self.generate_emission_prob()
        tables['spatialWeight'] = self.generate_spatial_weight()
        tables['intensityWeight'] = self.generate_intensity_weight()
        tables['textureWeight'] = self.generate_texture_weight()
        tables['positionBias'] = self.generate_position_bias()
        tables['bucketBias'] = self.generate_bucket_bias()
        tables['textureBias'] = self.generate_texture_bias()
        
        # Save all tables
        print("\n" + "=" * 70)
        print("SAVING TABLES")
        print("=" * 70)
        
        for name, data in tables.items():
            filename = os.path.join(output_dir, f'{name}.npy')
            np.save(filename, data)
            print(f"  ✓ Saved {filename} ({data.nbytes / 1024:.2f} KB)")
        
        # Calculate total memory
        total_memory = sum(t.nbytes for t in tables.values())
        
        # Compare to dense transition matrix
        dense_size = self.total_states ** 2 * 8
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total states: {self.total_states:,}")
        print(f"Total table memory: {total_memory / (1024**2):.2f} MB")
        print(f"Dense transition matrix would be: {dense_size / (1024**3):.2f} GB")
        print(f"Memory savings: {(1 - total_memory / dense_size) * 100:.2f}%")
        print(f"Compression ratio: {dense_size / total_memory:.1f}×")
        
        # Analyze sparsity
        print("\nSparsity Analysis:")
        spatial_nnz = np.count_nonzero(tables['spatialWeight'])
        spatial_size = tables['spatialWeight'].size
        spatial_sparsity = (1 - spatial_nnz / spatial_size) * 100
        print(f"  Spatial transitions: {spatial_sparsity:.4f}% sparse")
        print(f"  Avg predecessors/state: ~{spatial_nnz / self.num_positions:.1f}")
        
        print("\n✅ All tables generated successfully!")
        print("=" * 70)
        
        return tables


def main():
    """Main entry point"""
    
    print("\n🔥 COMPREHENSIVE TRELLIS TABLE GENERATOR 🔥\n")
    
    # Check for configuration argument
    if len(sys.argv) > 1:
        config_name = sys.argv[1].upper()
        if config_name not in CONFIGS:
            print(f"❌ Unknown configuration: {config_name}")
            print(f"Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)
    else:
        # Show available configurations
        print("Available configurations:")
        for i, name in enumerate(CONFIGS.keys(), 1):
            config = CONFIGS[name]
            total_states = (config['grid_width'] * config['grid_height'] * 
                          config['num_buckets'] * config['num_textures'])
            print(f"  {i}. {name}: {total_states:,} states")
        
        choice = input("\nSelect configuration (name or number): ").strip()
        
        try:
            if choice.isdigit():
                config_name = list(CONFIGS.keys())[int(choice) - 1]
            else:
                config_name = choice.upper()
        except:
            print("Invalid choice, using SMALL")
            config_name = 'SMALL'
    
    if config_name not in CONFIGS:
        print(f"❌ Configuration not found: {config_name}")
        sys.exit(1)
    
    config = CONFIGS[config_name]
    
    # Create output directory
    output_dir = f"models/{config_name.lower()}/tables"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Generate tables
    generator = TableGenerator(config)
    
    import time
    start_time = time.time()
    
    tables = generator.generate_all(output_dir)
    
    elapsed = time.time() - start_time
    
    print(f"\n⏱ Total generation time: {elapsed:.2f} seconds")
    
    # Save metadata
    metadata = {
        'config_name': config_name,
        'config': config,
        'total_states': generator.total_states,
        'generation_time': elapsed,
        'tables': {name: {'shape': list(data.shape), 'size_bytes': int(data.nbytes)}
                   for name, data in tables.items()}
    }
    
    import json
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to: {metadata_file}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"1. Ensure the Trellis model is compiled:")
    print(f"   trellis image_hmm.trellis")
    print(f"\n2. Run the model with these tables:")
    print(f"   python runner.py {config_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
