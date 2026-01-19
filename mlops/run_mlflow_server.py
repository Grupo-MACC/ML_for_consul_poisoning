"""
MLflow Server Launcher for Consul Poisoning Detection Project
Run this script to start the MLflow UI
"""
import os
import sys
import subprocess
from pathlib import Path

# Get project root
MLOPS_DIR = Path(__file__).parent
PROJECT_ROOT = MLOPS_DIR.parent
MLRUNS_DIR = MLOPS_DIR / "mlruns"

# Create mlruns directory if it doesn't exist
MLRUNS_DIR.mkdir(exist_ok=True)

def main():
    """Start MLflow server with UI."""
    print("\n" + "="*60)
    print("🚀 Starting MLflow Server")
    print("="*60)
    print(f"📂 Tracking Directory: {MLRUNS_DIR}")
    print(f"🌐 UI will be available at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start MLflow server
    cmd = [
        sys.executable, "-m", "mlflow", "ui",
        "--backend-store-uri", f"file://{MLRUNS_DIR.absolute()}",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n\n👋 MLflow server stopped.")


if __name__ == "__main__":
    main()
