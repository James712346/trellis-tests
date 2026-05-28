#!/usr/bin/env python3
"""
Trellis HMM runner

Usage:
    python runner.py --configs SMALL,MEDIUM [--mode forward|viterbi|both]
        [--images test_real.jpg,test_4k.jpg] [--log-gpu] [--log-cpu] 
        [--log-interval 1.0] [--db-path PATH] [--no-warmup]
"""

import argparse
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

# Optional psutil for CPU metrics
try:
    import psutil
except Exception:
    psutil = None

# Globals
TRELLIS_AVAILABLE = False
HMM = None


# --- Enhanced Metrics Database ------------------------------------------------
class EnhancedMetricsDB:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._ensure_db_dir()
        self._connect()
        self._create_schema()

    def _ensure_db_dir(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self):
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

    def _create_schema(self):
        c = self.conn.cursor()
        
        # Enhanced runs table with detailed metadata
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_uuid TEXT UNIQUE NOT NULL,
                config_name TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT,
                duration_seconds REAL,
                mode TEXT,
                status TEXT DEFAULT 'running',
                error_message TEXT,
                total_states INTEGER,
                grid_width INTEGER,
                grid_height INTEGER,
                num_buckets INTEGER,
                num_textures INTEGER,
                warmup_enabled INTEGER,
                notes TEXT
            )
        """)
        
        # Image test sessions
        c.execute("""
            CREATE TABLE IF NOT EXISTS test_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                image_name TEXT NOT NULL,
                image_path TEXT,
                image_width INTEGER,
                image_height INTEGER,
                num_sequences INTEGER,
                sequence_method TEXT,
                start_ts TEXT NOT NULL,
                end_ts TEXT,
                duration_seconds REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)
        
        # Algorithm execution details
        c.execute("""
            CREATE TABLE IF NOT EXISTS algorithm_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_session_id INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                start_ts TEXT NOT NULL,
                end_ts TEXT,
                duration_seconds REAL,
                num_sequences INTEGER,
                num_parallel INTEGER,
                sequences_per_sec REAL,
                warmup_time_seconds REAL,
                status TEXT DEFAULT 'running',
                error_message TEXT,
                FOREIGN KEY (test_session_id) REFERENCES test_sessions(id)
            )
        """)
        
        # GPU metrics
        c.execute("""
            CREATE TABLE IF NOT EXISTS gpu_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                run_id INTEGER,
                test_session_id INTEGER,
                algorithm_run_id INTEGER,
                gpu_index INTEGER NOT NULL,
                temperature_c INTEGER,
                utilization_pct INTEGER,
                memory_total_mb INTEGER,
                memory_used_mb INTEGER,
                memory_free_mb INTEGER,
                power_draw_w REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (test_session_id) REFERENCES test_sessions(id),
                FOREIGN KEY (algorithm_run_id) REFERENCES algorithm_runs(id)
            )
        """)
        
        # CPU metrics
        c.execute("""
            CREATE TABLE IF NOT EXISTS cpu_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                run_id INTEGER,
                test_session_id INTEGER,
                algorithm_run_id INTEGER,
                cpu_percent REAL,
                mem_total_mb INTEGER,
                mem_used_mb INTEGER,
                mem_available_mb INTEGER,
                mem_percent REAL,
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (test_session_id) REFERENCES test_sessions(id),
                FOREIGN KEY (algorithm_run_id) REFERENCES algorithm_runs(id)
            )
        """)
        
        # Create indexes for efficient queries
        c.execute("CREATE INDEX IF NOT EXISTS idx_gpu_metrics_ts ON gpu_metrics(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_gpu_metrics_run ON gpu_metrics(run_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cpu_metrics_ts ON cpu_metrics(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_cpu_metrics_run ON cpu_metrics(run_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_algorithm_runs_session ON algorithm_runs(test_session_id)")
        
        self.conn.commit()

    def create_run(self, run_uuid: str, config_name: str, mode: str, 
                   total_states: int, grid_width: int, grid_height: int,
                   num_buckets: int, num_textures: int, warmup_enabled: bool,
                   notes: str = None) -> int:
        c = self.conn.cursor()
        start_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c.execute("""
            INSERT INTO runs (run_uuid, config_name, start_ts, mode, total_states,
                            grid_width, grid_height, num_buckets, num_textures,
                            warmup_enabled, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_uuid, config_name, start_ts, mode, total_states,
              grid_width, grid_height, num_buckets, num_textures,
              int(warmup_enabled), notes))
        self.conn.commit()
        return c.lastrowid

    def complete_run(self, run_id: int, status: str = 'completed', error_message: str = None):
        end_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c = self.conn.cursor()
        # Calculate duration
        c.execute("SELECT start_ts FROM runs WHERE id=?", (run_id,))
        row = c.fetchone()
        if row:
            start = datetime.fromisoformat(row[0])
            end = datetime.fromisoformat(end_ts)
            duration = (end - start).total_seconds()
            c.execute("""
                UPDATE runs SET end_ts=?, duration_seconds=?, status=?, error_message=?
                WHERE id=?
            """, (end_ts, duration, status, error_message, run_id))
            self.conn.commit()

    def create_test_session(self, run_id: int, image_name: str, image_path: str,
                           image_width: int, image_height: int, num_sequences: int,
                           sequence_method: str) -> int:
        c = self.conn.cursor()
        start_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c.execute("""
            INSERT INTO test_sessions (run_id, image_name, image_path, image_width,
                                      image_height, num_sequences, sequence_method, start_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (run_id, image_name, image_path, image_width, image_height,
              num_sequences, sequence_method, start_ts))
        self.conn.commit()
        return c.lastrowid

    def complete_test_session(self, session_id: int):
        end_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c = self.conn.cursor()
        c.execute("SELECT start_ts FROM test_sessions WHERE id=?", (session_id,))
        row = c.fetchone()
        if row:
            start = datetime.fromisoformat(row[0])
            end = datetime.fromisoformat(end_ts)
            duration = (end - start).total_seconds()
            c.execute("""
                UPDATE test_sessions SET end_ts=?, duration_seconds=?
                WHERE id=?
            """, (end_ts, duration, session_id))
            self.conn.commit()

    def create_algorithm_run(self, test_session_id: int, algorithm: str,
                           num_sequences: int, num_parallel: int = 1) -> int:
        c = self.conn.cursor()
        start_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c.execute("""
            INSERT INTO algorithm_runs (test_session_id, algorithm, start_ts,
                                       num_sequences, num_parallel)
            VALUES (?, ?, ?, ?, ?)
        """, (test_session_id, algorithm, start_ts, num_sequences, num_parallel))
        self.conn.commit()
        return c.lastrowid

    def complete_algorithm_run(self, algo_run_id: int, warmup_time: float = None,
                              status: str = 'completed', error_message: str = None):
        end_ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
        c = self.conn.cursor()
        c.execute("SELECT start_ts, num_sequences FROM algorithm_runs WHERE id=?", (algo_run_id,))
        row = c.fetchone()
        if row:
            start = datetime.fromisoformat(row[0])
            end = datetime.fromisoformat(end_ts)
            duration = (end - start).total_seconds()
            num_seqs = row[1]
            seqs_per_sec = num_seqs / duration if duration > 0 else 0
            c.execute("""
                UPDATE algorithm_runs 
                SET end_ts=?, duration_seconds=?, sequences_per_sec=?,
                    warmup_time_seconds=?, status=?, error_message=?
                WHERE id=?
            """, (end_ts, duration, seqs_per_sec, warmup_time, status, error_message, algo_run_id))
            self.conn.commit()

    def insert_gpu_metric(self, ts: str, gpu_index: int, temp_c: Optional[int],
                          util_pct: Optional[int], mem_total_mb: Optional[int],
                          mem_used_mb: Optional[int], power_draw_w: Optional[float] = None,
                          run_id: int = None, test_session_id: int = None,
                          algorithm_run_id: int = None):
        mem_free = mem_total_mb - mem_used_mb if (mem_total_mb and mem_used_mb) else None
        self.conn.execute("""
            INSERT INTO gpu_metrics (ts, run_id, test_session_id, algorithm_run_id,
                                    gpu_index, temperature_c, utilization_pct,
                                    memory_total_mb, memory_used_mb, memory_free_mb,
                                    power_draw_w)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, run_id, test_session_id, algorithm_run_id, gpu_index,
              temp_c, util_pct, mem_total_mb, mem_used_mb, mem_free, power_draw_w))

    def insert_cpu_metric(self, ts: str, cpu_percent: float,
                          mem_total_mb: Optional[int], mem_used_mb: Optional[int],
                          mem_available_mb: Optional[int] = None,
                          mem_percent: Optional[float] = None,
                          run_id: int = None, test_session_id: int = None,
                          algorithm_run_id: int = None):
        self.conn.execute("""
            INSERT INTO cpu_metrics (ts, run_id, test_session_id, algorithm_run_id,
                                    cpu_percent, mem_total_mb, mem_used_mb,
                                    mem_available_mb, mem_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ts, run_id, test_session_id, algorithm_run_id, cpu_percent,
              mem_total_mb, mem_used_mb, mem_available_mb, mem_percent))

    def commit(self):
        self.conn.commit()

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


# --- Enhanced GPU/CPU Logger -------------------------------------------------
class EnhancedGpuCpuLogger(threading.Thread):
    """Background thread that polls nvidia-smi and psutil with context awareness."""

    def __init__(self, db: EnhancedMetricsDB, interval: float = 1.0,
                 stop_event: Optional[threading.Event] = None):
        super().__init__(daemon=True)
        self.db = db
        self.interval = max(0.1, float(interval))
        self.stop_event = stop_event or threading.Event()
        self._nvsmi_available = self._check_nvidia_smi()
        
        # Context tracking
        self.current_run_id = None
        self.current_session_id = None
        self.current_algo_run_id = None
        self._context_lock = threading.Lock()

    def set_context(self, run_id=None, session_id=None, algo_run_id=None):
        """Update the current context for metric correlation."""
        with self._context_lock:
            if run_id is not None:
                self.current_run_id = run_id
            if session_id is not None:
                self.current_session_id = session_id
            if algo_run_id is not None:
                self.current_algo_run_id = algo_run_id

    def clear_algo_context(self):
        """Clear algorithm context but keep run/session."""
        with self._context_lock:
            self.current_algo_run_id = None

    def _check_nvidia_smi(self) -> bool:
        try:
            subprocess.run(["nvidia-smi", "--help"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception:
            return False

    def _poll_nvidia_smi(self):
        """Poll nvidia-smi for GPU metrics including power draw."""
        if not self._nvsmi_available:
            return []

        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,utilization.gpu,memory.total,memory.used,power.draw",
                "--format=csv,noheader,nounits"
            ]
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            lines = [line.strip() for line in out.splitlines() if line.strip()]
            rows = []
            for ln in lines:
                parts = [p.strip() for p in ln.split(',')]
                if len(parts) >= 5:
                    gpu_index = int(parts[0])
                    temp = int(parts[1]) if parts[1] != 'N/A' else None
                    util = int(parts[2]) if parts[2] != 'N/A' else None
                    mem_total = int(parts[3]) if parts[3] != 'N/A' else None
                    mem_used = int(parts[4]) if parts[4] != 'N/A' else None
                    power = float(parts[5]) if len(parts) > 5 and parts[5] != 'N/A' else None
                    rows.append((gpu_index, temp, util, mem_total, mem_used, power))
            return rows
        except Exception:
            return []

    def _poll_cpu(self):
        if psutil is None:
            return None
        try:
            cpu_pct = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            mem_total_mb = int(vm.total / (1024 * 1024))
            mem_used_mb = int((vm.total - vm.available) / (1024 * 1024))
            mem_available_mb = int(vm.available / (1024 * 1024))
            mem_percent = vm.percent
            return cpu_pct, mem_total_mb, mem_used_mb, mem_available_mb, mem_percent
        except Exception:
            return None

    def run(self):
        while not self.stop_event.is_set():
            ts = datetime.now(UTC).isoformat(sep=" ", timespec="milliseconds")
            
            with self._context_lock:
                run_id = self.current_run_id
                session_id = self.current_session_id
                algo_run_id = self.current_algo_run_id
            
            # Poll GPU
            gpu_rows = self._poll_nvidia_smi()
            for (gpu_index, temp, util, mem_total, mem_used, power) in gpu_rows:
                try:
                    self.db.insert_gpu_metric(ts, gpu_index, temp, util, mem_total,
                                            mem_used, power, run_id, session_id,
                                            algo_run_id)
                except Exception:
                    pass

            # Poll CPU
            cpu_sample = self._poll_cpu()
            if cpu_sample is not None:
                cpu_pct, mem_total_mb, mem_used_mb, mem_avail_mb, mem_pct = cpu_sample
                try:
                    self.db.insert_cpu_metric(ts, cpu_pct, mem_total_mb, mem_used_mb,
                                            mem_avail_mb, mem_pct, run_id, session_id,
                                            algo_run_id)
                except Exception:
                    pass

            try:
                self.db.commit()
            except Exception:
                pass

            # Interruptible wait
            stop_wait_until = time.time() + self.interval
            while time.time() < stop_wait_until and not self.stop_event.is_set():
                time.sleep(0.05)


# --- Enhanced Trellis Runner -------------------------------------------------
class EnhancedTrellisRunner:
    """Enhanced runner with detailed logging and multi-config support."""

    def __init__(self, config_name: str, tables_dir: str, db: EnhancedMetricsDB,
                 logger: EnhancedGpuCpuLogger, mode: str, warmup: bool, quiet: bool = False):
        self.config_name = config_name
        self.tables_dir = Path(tables_dir)
        self.db = db
        self.logger = logger
        self.mode = mode
        self.warmup = warmup
        self.quiet = quiet
        self.run_id = None
        
        self._load_metadata_defaults()
        self.hmm = None

        if not quiet:
            self._print_header()

    def _load_metadata_defaults(self):
        metadata_file = self.tables_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                try:
                    self.metadata = json.load(f)
                    self.config = self.metadata.get("config", {})
                except Exception:
                    self.metadata = {}
                    self.config = {}
        else:
            self.metadata = {}
            self.config = {}

        self.config.setdefault("grid_width", 48)
        self.config.setdefault("grid_height", 27)
        self.config.setdefault("num_buckets", 8)
        self.config.setdefault("num_textures", 4)

        self.grid_width = int(self.config["grid_width"])
        self.grid_height = int(self.config["grid_height"])
        self.num_buckets = int(self.config["num_buckets"])
        self.num_textures = int(self.config["num_textures"])

        self.num_positions = self.grid_width * self.grid_height
        self.total_states = self.num_positions * self.num_buckets * self.num_textures

    def _print_header(self):
        print("=" * 80)
        print(f"ENHANCED TRELLIS HMM RUNNER - {self.config_name}")
        print("=" * 80)
        print(f"Grid: {self.grid_width}x{self.grid_height} = {self.num_positions:,} positions")
        print(f"Buckets: {self.num_buckets}, Textures: {self.num_textures}")
        print(f"Total states: {self.total_states:,}")
        print(f"Mode: {self.mode}, Warmup: {self.warmup}")
        print("=" * 80)

    def load_tables(self) -> bool:
        """Load required numpy tables."""
        if not self.tables_dir.exists():
            if not self.quiet:
                print(f"ERROR: tables directory not found: {self.tables_dir}")
            return False

        table_files = {
            "initialProb": "initialProb.npy",
            "emissionProb": "emissionProb.npy",
            "spatialWeight": "spatialWeight.npy",
            "intensityWeight": "intensityWeight.npy",
            "textureWeight": "textureWeight.npy",
            "positionBias": "positionBias.npy",
            "bucketBias": "bucketBias.npy",
            "textureBias": "textureBias.npy",
        }

        self.tables = {}
        for name, fname in table_files.items():
            p = self.tables_dir / fname
            if not p.exists():
                if not self.quiet:
                    print(f"ERROR: required table file missing: {p}")
                return False
            arr = np.load(str(p))
            self.tables[name] = arr

        return True

    def initialize_hmm(self) -> bool:
        """Instantiate the HMM."""
        global TRELLIS_AVAILABLE, HMM
        if not TRELLIS_AVAILABLE or HMM is None:
            if not self.quiet:
                print("ERROR: Trellis HMM module not available.")
            return False

        try:
            self.hmm = HMM(self.tables)
            return True
        except Exception as e:
            if not self.quiet:
                print(f"ERROR initializing HMM: {e}")
            return False

    def start_run(self, notes: str = None):
        """Start a new run and register with DB."""
        import uuid
        run_uuid = str(uuid.uuid4())
        self.run_id = self.db.create_run(
            run_uuid, self.config_name, self.mode, self.total_states,
            self.grid_width, self.grid_height, self.num_buckets,
            self.num_textures, self.warmup, notes
        )
        self.logger.set_context(run_id=self.run_id)
        if not self.quiet:
            print(f"Run started: ID={self.run_id}, UUID={run_uuid}")

    def complete_run(self, status='completed', error=None):
        """Mark run as complete."""
        if self.run_id:
            self.db.complete_run(self.run_id, status, error)

    def run_on_image(self, image_name: str, image_array: np.ndarray,
                    image_path: str, method: str = "rows",
                    max_sequences: int = None):
        """Run algorithms on a single image with full tracking."""
        h, w = image_array.shape
        sequences = self._create_sequences(image_array, method, max_sequences)
        
        session_id = self.db.create_test_session(
            self.run_id, image_name, image_path, w, h,
            len(sequences), method
        )
        self.logger.set_context(session_id=session_id)
        
        if not self.quiet:
            print(f"\n{'='*60}")
            print(f"Image: {image_name} ({w}x{h}), Sequences: {len(sequences)}")
        
        try:
            if self.mode in ("forward", "both"):
                self._run_forward(session_id, sequences)
            
            if self.mode in ("viterbi", "both"):
                self._run_viterbi(session_id, sequences)
                
            self.db.complete_test_session(session_id)
        except Exception as e:
            if not self.quiet:
                print(f"ERROR in test session: {e}")
                traceback.print_exc()

    def _create_sequences(self, image_array, method, max_sequences):
        """Extract sequences from image."""
        h, w = image_array.shape
        sequences = []
        if method == "rows":
            indices = range(h)
            if max_sequences and max_sequences < h:
                step = max(1, h // max_sequences)
                indices = range(0, h, step)
            for i in indices:
                sequences.append(image_array[i, :].tolist())
        elif method == "cols":
            indices = range(w)
            if max_sequences and max_sequences < w:
                step = max(1, w // max_sequences)
                indices = range(0, w, step)
            for i in indices:
                sequences.append(image_array[:, i].tolist())
        return sequences

    def _run_forward(self, session_id: int, sequences: List):
        """Run forward algorithm with tracking."""
        algo_run_id = self.db.create_algorithm_run(session_id, "forward", len(sequences))
        self.logger.set_context(algo_run_id=algo_run_id)
        
        warmup_time = 0
        try:
            if self.warmup:
                wu_start = time.time()
                self.hmm.forward(sequences[:min(5, len(sequences))])
                warmup_time = time.time() - wu_start
            
            log_probs = self.hmm.forward(sequences)
            self.db.complete_algorithm_run(algo_run_id, warmup_time, 'completed')
            
            if not self.quiet:
                print(f"  Forward: {len(sequences)} sequences completed")
                
        except Exception as e:
            self.db.complete_algorithm_run(algo_run_id, warmup_time, 'failed', str(e))
            if not self.quiet:
                print(f"  Forward FAILED: {e}")
        finally:
            self.logger.clear_algo_context()

    def _run_viterbi(self, session_id: int, sequences: List):
        """Run viterbi algorithm with tracking."""
        # Determine parallelism
        if self.total_states < 20000:
            num_parallel = 13
        elif self.total_states < 50000:
            num_parallel = 9
        else:
            num_parallel = 7
            
        algo_run_id = self.db.create_algorithm_run(
            session_id, "viterbi", len(sequences), num_parallel
        )
        self.logger.set_context(algo_run_id=algo_run_id)
        
        warmup_time = 0
        try:
            if self.warmup:
                wu_start = time.time()
                self.hmm.viterbi(sequences[:min(3, len(sequences))], num_parallel=1)
                warmup_time = time.time() - wu_start
            
            states = self.hmm.viterbi(sequences, num_parallel=num_parallel)
            self.db.complete_algorithm_run(algo_run_id, warmup_time, 'completed')
            
            if not self.quiet:
                print(f"  Viterbi: {len(sequences)} sequences, parallel={num_parallel}")
                
        except Exception as e:
            self.db.complete_algorithm_run(algo_run_id, warmup_time, 'failed', str(e))
            if not self.quiet:
                print(f"  Viterbi FAILED: {e}")
        finally:
            self.logger.clear_algo_context()


# --- Utilities ----------------------------------------------------------------
def import_trellis_module(config_name: str) -> bool:
    """Import trellis module from models/{config}/trellis.py."""
    global TRELLIS_AVAILABLE, HMM
    trellis_path = Path(f"trellis.py")
    if not trellis_path.exists():
        TRELLIS_AVAILABLE = False
        return False
    try:
        spec = importlib.util.spec_from_file_location("trellis", str(trellis_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        HMM = getattr(module, "HMM", None)
        TRELLIS_AVAILABLE = HMM is not None
        return TRELLIS_AVAILABLE
    except Exception as e:
        print(f"Error importing trellis for {config_name}: {e}")
        TRELLIS_AVAILABLE = False
        return False


def find_available_images(image_names: List[str] = None) -> Dict[str, Path]:
    """Find available test images."""
    search_dirs = ["../../images", "../images", "images", "."]
    
    if image_names is None:
        # Default search patterns
        patterns = ["test_*.jpg", "test_*.png"]
    else:
        patterns = image_names
    
    found = {}
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for pattern in patterns:
            if '*' in pattern:
                import glob
                matches = glob.glob(os.path.join(base_dir, pattern))
                for match in matches:
                    p = Path(match)
                    if p.exists() and p.name not in found:
                        found[p.name] = p
            else:
                p = Path(base_dir) / pattern
                if p.exists() and p.name not in found:
                    found[p.name] = p
    
    return found


# --- Main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Trellis HMM runner with comprehensive logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single config with all available images
  python %(prog)s --configs SMALL --log-gpu --log-cpu
  
  # Run multiple configs with specific images
  python %(prog)s --configs SMALL,MEDIUM,LARGE --images test_real.jpg,test_4k.jpg
  
  # Run forward only on specific images
  python %(prog)s --configs SMALL --mode forward --images test_real.jpg --no-warmup
        """
    )
    parser.add_argument("--configs", required=True, 
                       help="Comma-separated list of config names (e.g., SMALL,MEDIUM,LARGE)")
    parser.add_argument("--mode", choices=["forward", "viterbi", "both"], 
                       default="both", help="Which algorithm(s) to run")
    parser.add_argument("--images", type=str, default=None,
                       help="Comma-separated list of image filenames or patterns (e.g., test_real.jpg,test_*.png)")
    parser.add_argument("--sequence-method", choices=["rows", "cols"], default="rows",
                       help="How to extract sequences from images")
    parser.add_argument("--sequences-per-image", type=int, default=None,
                       help="Limit number of sequences extracted per image")
    parser.add_argument("--log-gpu", action="store_true", 
                       help="Enable GPU metrics logging")
    parser.add_argument("--log-cpu", action="store_true",
                       help="Enable CPU metrics logging (requires psutil)")
    parser.add_argument("--log-interval", type=float, default=1.0,
                       help="Polling interval in seconds for metrics (default: 1.0)")
    parser.add_argument("--db-path", type=str, default=None,
                       help="Path to SQLite database (default: ./metrics.db)")
    parser.add_argument("--no-warmup", action="store_true",
                       help="Skip warmup runs")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimize printed output")
    parser.add_argument("--notes", type=str, default=None,
                       help="Optional notes about this run")
    args = parser.parse_args()

    # Parse configs
    config_names = [c.strip().upper() for c in args.configs.split(",")]
    
    # Parse images
    image_filter = None
    if args.images:
        image_filter = [i.strip() for i in args.images.split(",")]
    
    # Setup database
    db_path = Path(args.db_path) if args.db_path else Path("./metrics.db")
    if not args.quiet:
        print(f"Database: {db_path}")
    
    # Initialize metrics DB
    metrics_db = EnhancedMetricsDB(db_path)
    
    # Setup logger if requested
    logger_thread = None
    stop_event = threading.Event()
    
    if args.log_gpu or args.log_cpu:
        if args.log_cpu and psutil is None:
            print("WARNING: psutil not available, CPU logging disabled")
        logger_thread = EnhancedGpuCpuLogger(
            metrics_db, 
            interval=args.log_interval, 
            stop_event=stop_event
        )
        logger_thread.start()
        if not args.quiet:
            print(f"Metrics logger started (interval={args.log_interval}s)")
    else:
        # Create a dummy logger
        logger_thread = type('obj', (object,), {
            'set_context': lambda *args, **kwargs: None,
            'clear_algo_context': lambda: None
        })()
    
    warmup = not args.no_warmup
    
    try:
        # Run each configuration
        for config_name in config_names:
            if not args.quiet:
                print(f"\n{'#'*80}")
                print(f"# CONFIGURATION: {config_name}")
                print(f"{'#'*80}\n")
            
            # Check if config directory exists
            config_dir = Path(f"models/{config_name.lower()}")
            if not config_dir.exists():
                print(f"ERROR: Configuration directory not found: {config_dir}")
                continue
            os.chdir("/home/jamesp/Documents/Study/2025/EGH400/image-sparse")
            os.chdir(config_dir)
            # Import trellis module for this config
            if not import_trellis_module(config_name):
                print(f"ERROR: Could not import trellis module for {config_name}")
                continue
            
            # Setup tables directory
            tables_dir = Path("tables")
            if not tables_dir.exists():
                print(f"ERROR: Tables directory not found: {tables_dir}")
                continue
            
            # Create runner
            runner = EnhancedTrellisRunner(
                config_name, str(tables_dir), metrics_db, logger_thread,
                args.mode, warmup, args.quiet
            )
            
            # Load tables
            if not runner.load_tables():
                print(f"ERROR: Failed to load tables for {config_name}")
                continue
            
            # Initialize HMM
            if not runner.initialize_hmm():
                print(f"ERROR: Failed to initialize HMM for {config_name}")
                continue
            
            # Start run
            notes = args.notes or f"Multi-config run: {','.join(config_names)}"
            runner.start_run(notes)
            
            # Find available images
            available_images = find_available_images(image_filter)
            
            if not available_images:
                print("WARNING: No test images found, creating synthetic sequences")
                # Create synthetic data
                rng = np.random.RandomState(12345)
                synthetic = rng.randint(0, 256, size=(100, 200), dtype=np.uint8)
                runner.run_on_image(
                    "synthetic", synthetic, "synthetic",
                    method=args.sequence_method,
                    max_sequences=args.sequences_per_image
                )
            else:
                # Run on each image
                for img_name, img_path in sorted(available_images.items()):
                    try:
                        if not args.quiet:
                            print(f"\nLoading image: {img_path}")
                        img = Image.open(img_path).convert("L")
                        img_array = np.array(img)
                        
                        runner.run_on_image(
                            img_name, img_array, str(img_path),
                            method=args.sequence_method,
                            max_sequences=args.sequences_per_image
                        )
                    except Exception as e:
                        print(f"ERROR processing image {img_name}: {e}")
                        traceback.print_exc()
            
            # Complete run
            runner.complete_run('completed')
            
            if not args.quiet:
                print(f"\n{config_name} run completed")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        # Stop logger
        if hasattr(logger_thread, 'stop_event'):
            stop_event.set()
            if hasattr(logger_thread, 'join'):
                logger_thread.join(timeout=5.0)
        
        # Close database
        metrics_db.close()
        
        if not args.quiet:
            print(f"\nMetrics saved to: {db_path}")
            print("\nDone!")


if __name__ == "__main__":
    main()
