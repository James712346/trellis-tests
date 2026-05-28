#!/usr/bin/env python3
"""
Massive Trellis Configuration Generator
Creates multiple .trellis files with increasing complexity
Shows compilation times, state spaces, and sparsity patterns
"""

import os
import time
import subprocess
import json
import sys
# Configuration presets (increasingly painful)
CONFIGS = {
    'TINY': {
        'grid_width': 16,
        'grid_height': 9,
        'num_buckets': 4,
        'num_textures': 2,
        'num_cases': 7,
        'max_spatial_dist': 1,
        'max_intensity_diff': 1,
        'max_texture_diff': 1,
        'description': 'Quick test configuration'
    },
    'SMALL': {
        'grid_width': 32,
        'grid_height': 18,
        'num_buckets': 8,
        'num_textures': 4,
        'num_cases': 9,
        'max_spatial_dist': 1,
        'max_intensity_diff': 1,
        'max_texture_diff': 1,
        'description': 'Reasonable for initial testing - Ultra sparse'
    },
    'MEDIUM': {
        'grid_width': 48,
        'grid_height': 27,
        'num_buckets': 8,
        'num_textures': 4,
        'num_cases': 9,
        'max_spatial_dist': 1,
        'max_intensity_diff': 1,
        'max_texture_diff': 1,
        'description': 'Medium complexity, ultra sparse (95%+)'
    },
    'LARGE': {
        'grid_width': 64,
        'grid_height': 36,
        'num_buckets': 8,
        'num_textures': 4,
        'num_cases': 13,
        'max_spatial_dist': 1,
        'max_intensity_diff': 1,
        'max_texture_diff': 1,
        'description': 'Large state space, ultra sparse'
    },
    'XLARGE': {
        'grid_width': 96,
        'grid_height': 54,
        'num_buckets': 16,
        'num_textures': 8,
        'num_cases': 13,
        'max_spatial_dist': 1,
        'max_intensity_diff': 1,
        'max_texture_diff': 1,
        'description': 'Extra large, ultra sparse (>99%)'
    },
    'MASSIVE': {
        'grid_width': 128,
        'grid_height': 72,
        'num_buckets': 16,
        'num_textures': 8,
        'num_cases': 17,
        'max_spatial_dist': 9999,
        'max_intensity_diff': 9999,
        'max_texture_diff': 9999,
        'description': 'Massive state space - maximum sparsity!'
    }
}


def calculate_stats(config):
    """Calculate state space statistics"""
    w = config['grid_width']
    h = config['grid_height']
    b = config['num_buckets']
    t = config['num_textures']
    
    # Get sparsity constraints
    max_spatial = config.get('max_spatial_dist', 1)
    max_intensity = config.get('max_intensity_diff', 1)
    max_texture = config.get('max_texture_diff', 1)
    
    num_positions = w * h
    total_states = num_positions * b * t
    
    # Calculate actual predecessors based on constraints
    # Spatial: 1 stay + up to 4 adjacent (dist=1) = ~5 on average (less at boundaries)
    avg_spatial_pred = min(5, 1 + 4 * (max_spatial >= 1) + 4 * (max_spatial >= 2))
    
    # Intensity: 1 stay + 2*max_intensity_diff neighbors
    avg_intensity_pred = 1 + 2 * max_intensity
    
    # Texture: 1 stay + 2*max_texture_diff neighbors  
    avg_texture_pred = 1 + 2 * max_texture
    
    # For ultra-sparse: only allow ONE component to change at a time
    # This gives us: spatial OR intensity OR texture changes, not combinations
    avg_predecessors = avg_spatial_pred + avg_intensity_pred + avg_texture_pred - 2
    # (subtract 2 because "stay" is counted 3 times)
    
    # Dense transition matrix size
    dense_entries = total_states ** 2
    dense_size_gb = (dense_entries * 8) / (1024**3)
    
    # Sparse transition matrix (estimated)
    sparse_entries = total_states * avg_predecessors
    sparsity = (1 - sparse_entries / dense_entries) * 100
    
    return {
        'num_positions': num_positions,
        'total_states': total_states,
        'dense_entries': dense_entries,
        'dense_size_gb': dense_size_gb,
        'sparse_entries': sparse_entries,
        'sparsity': sparsity,
        'avg_predecessors': avg_predecessors,
        'max_spatial_dist': max_spatial,
        'max_intensity_diff': max_intensity,
        'max_texture_diff': max_texture
    }


def generate_trellis_model(config, output_file):
    """Generate a Trellis model file with given configuration"""
    
    w = config['grid_width']
    h = config['grid_height']
    b = config['num_buckets']
    t = config['num_textures']
    num_cases = config['num_cases']
    
    stats = calculate_stats(config)
    
    # Generate the header
    header = f"""-- Trellis Sparse HMM for Image Processing
-- Configuration: {config['description']}
-- Generated configuration with {stats['total_states']:,} states
-- Grid: {w}×{h}, Buckets: {b}, Textures: {t}
-- Dense matrix would be: {stats['dense_size_gb']:.2f} GB
-- Sparse with ~{stats['avg_predecessors']} predecessors/state: {stats['sparsity']:.4f}% sparse

let gridWidth = {w}
let gridHeight = {h}
let numBuckets = {b}
let numTextures = {t}

-- Type definitions
alias Position = 0 .. {stats['num_positions'] - 1}
alias IntensityBucket = 0 .. {b - 1}
alias TextureType = 0 .. {t - 1}
alias Intensity = 0 .. 255

model """
    
    # Start model block - NOTE: Single brace, not double!
    model_body = """{
  -- Compound state: (spatial position, intensity bucket, texture type)
  state = (Position, IntensityBucket, TextureType)
  
  -- Observation: pixel intensity value
  output = Intensity

  -- Probability tables (provided at runtime)
  table initialProb(Position, IntensityBucket, TextureType)
  table emissionProb(Intensity, IntensityBucket)
  table spatialWeight(Position, Position)
  table intensityWeight(IntensityBucket, IntensityBucket)
  table textureWeight(TextureType, TextureType)
  
  -- Additional transition modulation tables
  table positionBias(Position)
  table bucketBias(IntensityBucket)
  table textureBias(TextureType)

  -- Initial state probability
  P(initial x) = initialProb(x[0], x[1], x[2])

  -- Emission probability (mainly depends on intensity bucket)
  P(output o | x) = emissionProb(o, x[1])

  -- Sparse transition probability
  -- Using set constraints to define sparse patterns
  P(transition x y) = """
    
    # Start transition cases - NOTE: Single brace!
    transitions = "{\n"
    
    case_num = 1
    
    # Case 1: Stay in same state
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Stay in same state (all components identical)
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * intensityWeight(x[1], y[1]) * textureWeight(x[2], y[2]) * \n"
        transitions += "      positionBias(x[0]) * bucketBias(x[1]) * textureBias(x[2])\n\n"
        case_num += 1
    
    # Case 2: Move right
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move right (+1 in position), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.85 * positionBias(y[0])\n\n"
        case_num += 1
    
    # Case 3: Move left
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move left (-1 in position), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.85 * positionBias(y[0])\n\n"
        case_num += 1
    
    # Case 4: Move down
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move down (+gridWidth in position), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - " + str(w) + ", b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.80 * positionBias(y[0])\n\n"
        case_num += 1
    
    # Case 5: Move up
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move up (-gridWidth in position), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + " + str(w) + ", b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.80 * positionBias(y[0])\n\n"
        case_num += 1
    
    # Case 6: Intensity increases by 1
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Intensity increases by 1, same position and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2, b1 == b2 - 1, t1 == t2, b2 != 0 } =>\n"
        transitions += "      intensityWeight(x[1], y[1]) * 0.75 * bucketBias(y[1])\n\n"
        case_num += 1
    
    # Case 7: Intensity decreases by 1
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Intensity decreases by 1, same position and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2, b1 == b2 + 1, t1 == t2, b1 != " + str(b-1) + " } =>\n"
        transitions += "      intensityWeight(x[1], y[1]) * 0.75 * bucketBias(y[1])\n\n"
        case_num += 1
    
    # Case 8: Texture increases by 1
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Texture increases by 1, same position and intensity
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2, b1 == b2, t1 == t2 - 1, t2 != 0 } =>\n"
        transitions += "      textureWeight(x[2], y[2]) * 0.70 * textureBias(y[2])\n\n"
        case_num += 1
    
    # Case 9: Texture decreases by 1
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Texture decreases by 1, same position and intensity
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2, b1 == b2, t1 == t2 + 1, t1 != " + str(t-1) + " } =>\n"
        transitions += "      textureWeight(x[2], y[2]) * 0.70 * textureBias(y[2])\n\n"
        case_num += 1
    
    # Diagonal movements for larger configs
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Diagonal move (right+down), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - " + str(w) + " - 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.60 * positionBias(y[0])\n\n"
        case_num += 1
    
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Diagonal move (left+down), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - " + str(w) + " + 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.60 * positionBias(y[0])\n\n"
        case_num += 1
    
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Diagonal move (right+up), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + " + str(w) + " - 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.60 * positionBias(y[0])\n\n"
        case_num += 1
    
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Diagonal move (left+up), same intensity and texture
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + " + str(w) + " + 1, b1 == b2, t1 == t2 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * 0.60 * positionBias(y[0])\n\n"
        case_num += 1
    
    # Combined movements (move + intensity change)
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move right + intensity increase
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - 1, b1 == b2 - 1, t1 == t2, b2 != 0 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * intensityWeight(x[1], y[1]) * 0.50\n\n"
        case_num += 1
    
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move right + intensity decrease
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 - 1, b1 == b2 + 1, t1 == t2, b1 != " + str(b-1) + " } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * intensityWeight(x[1], y[1]) * 0.50\n\n"
        case_num += 1
    
    # Move left + intensity change
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move left + intensity increase
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + 1, b1 == b2 - 1, t1 == t2, b2 != 0 } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * intensityWeight(x[1], y[1]) * 0.50\n\n"
        case_num += 1
    
    if case_num <= num_cases:
        transitions += f"""    -- Case {case_num}: Move left + intensity decrease
    | """ + "{ (p1, b1, t1) -> (p2, b2, t2) | \n        p1 == p2 + 1, b1 == b2 + 1, t1 == t2, b1 != " + str(b-1) + " } =>\n"
        transitions += "      spatialWeight(x[0], y[0]) * intensityWeight(x[1], y[1]) * 0.50\n\n"
        case_num += 1
    
    # OPTIONAL: Add a warning if num_cases exceeds our ultra-sparse budget
    if case_num < num_cases:
        print(f"  ⚠ Warning: Requested {num_cases} cases but only generated {case_num-1}")
        print(f"    This maintains ultra-sparsity (>95%)")
        print(f"    To add more cases, increase max_spatial_dist, max_intensity_diff, or max_texture_diff")
    
    # Close the transition definition
    transitions += "  }\n"
    
    # Close the model
    model_end = "}\n"
    
    # Combine all parts
    full_model = header + model_body + transitions + model_end
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(full_model)
    
    return stats


def compile_trellis_model(model_file):
    """Attempt to compile a Trellis model and measure time"""
    print(f"\n  Compiling {model_file}...")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ['trellis', os.path.basename(model_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(model_file)
        )
        compile_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"  ✓ Compilation successful in {compile_time:.2f}s")
            return {'success': True, 'time': compile_time, 'error': None}
        else:
            print(f"  ✗ Compilation failed in {compile_time:.2f}s")
            print(f"    Error: {result.stderr[:200]}")
            return {'success': False, 'time': compile_time, 'error': result.stderr}
    
    except subprocess.TimeoutExpired:
        compile_time = time.time() - start_time
        print(f"  ✗ Compilation timed out after {compile_time:.2f}s")
        return {'success': False, 'time': compile_time, 'error': 'Timeout'}
    
    except FileNotFoundError:
        print(f"  ⚠ Trellis compiler not found")
        return {'success': False, 'time': 0, 'error': 'Compiler not found'}


def main():
    """Generate and optionally compile multiple Trellis configurations"""
    
    print("=" * 80)
    print("MASSIVE TRELLIS CONFIGURATION GENERATOR")
    print("Generates increasingly complex sparse HMM configurations")
    print("=" * 80)
    
    # Show all configurations
    if len(sys.argv) > 1:
        cn = sys.argv[1].upper()
        if cn not in CONFIGS:
            print(f"❌ Unknown configuration: {cn}")
            print(f"Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)
        selected_configs = [cn]
    else:
        print("\nAvailable configurations:")
        for i, (name, config) in enumerate(CONFIGS.items(), 1):
            stats = calculate_stats(config)
            print(f"\n{i}. {name}:")
            print(f"   Description: {config['description']}")
            print(f"   State space: {stats['total_states']:,} states")
            print(f"   Grid: {config['grid_width']}×{config['grid_height']}")
            print(f"   Buckets: {config['num_buckets']}, Textures: {config['num_textures']}")
            print(f"   Sparsity constraints: spatial≤{stats['max_spatial_dist']}, "
                  f"intensity≤{stats['max_intensity_diff']}, texture≤{stats['max_texture_diff']}")
            print(f"   Avg predecessors: ~{stats['avg_predecessors']:.1f}")
            print(f"   Dense matrix: {stats['dense_size_gb']:.2f} GB")
            print(f"   Estimated sparsity: {stats['sparsity']:.4f}%")
        
        # User selection
        print("\n" + "=" * 80)
        print("Select configuration(s) to generate:")
        print("  Enter numbers separated by spaces (e.g., '1 2 3')")
        print("  Or 'all' to generate all configurations")
        selection = input("Selection: ").strip()
        
        if selection.lower() == 'all':
            selected_configs = list(CONFIGS.keys())
        else:
            try:
                indices = [int(x) for x in selection.split()]
                selected_configs = [list(CONFIGS.keys())[i-1] for i in indices if 1 <= i <= len(CONFIGS)]
            except:
                print("Invalid selection, using TINY only")
                selected_configs = ['TINY']
    
    # Ask about compilation
    if len(sys.argv) > 2:
        compile_choice = sys.argv[2].lower()
    else:
        compile_choice = input("\nAttempt to compile models? (y/N): ").strip().lower()
    should_compile = compile_choice == 'y'
    
    # Generate models
    results = {}
    
    print("\n" + "=" * 80)
    print("GENERATING MODELS")
    print("=" * 80)
    
    for config_name in selected_configs:
        os.makedirs("models", exist_ok=True)
        os.makedirs(f"models/{config_name.lower()}", exist_ok=True)
        config = CONFIGS[config_name]
        output_file = f"./models/{config_name.lower()}/image_hmm.trellis"
        
        print(f"\n[{config_name}]")
        print(f"  Generating {output_file}...")
        
        stats = generate_trellis_model(config, output_file)
        
        print(f"  ✓ Model generated")
        print(f"    File: {output_file}")
        print(f"    States: {stats['total_states']:,}")
        print(f"    Cases: {config['num_cases']}")
        
        result = {
            'config': config,
            'stats': stats,
            'file': output_file,
            'compilation': None
        }
        
        if should_compile:
            result['compilation'] = compile_trellis_model(output_file)
        
        results[config_name] = result
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for config_name, result in results.items():
        stats = result['stats']
        comp = result['compilation']
        
        print(f"\n{config_name}:")
        print(f"  File: {result['file']}")
        print(f"  States: {stats['total_states']:,}")
        print(f"  Dense matrix: {stats['dense_size_gb']:.2f} GB")
        print(f"  Sparsity: {stats['sparsity']:.4f}%")
        
        if comp:
            if comp['success']:
                print(f"  Compilation: ✓ Success ({comp['time']:.2f}s)")
            else:
                print(f"  Compilation: ✗ Failed ({comp['time']:.2f}s)")
                if comp['error'] and comp['error'] != 'Compiler not found':
                    print(f"    Error: {comp['error'][:100]}...")
    
    # Save results to JSON
    results_file = 'trellis_generation_results.json'
    with open(results_file, 'w') as f:
        json_results = {}
        for name, result in results.items():
            json_results[name] = {
                'config': result['config'],
                'stats': result['stats'],
                'file': result['file'],
                'compilation': result['compilation']
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Review the generated .trellis files")
    print("  2. If compilation succeeded, generate tables:")
    print("     python generate_tables.py <config_name>")
    print("  3. Run the model:")
    print("     python trellis_runner.py <config_name>")
    print("=" * 80)


if __name__ == "__main__":
    main()
