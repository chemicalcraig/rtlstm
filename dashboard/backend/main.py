import os
import json
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUTS_DIR = BASE_DIR / "inputs"
SRC_DIR = BASE_DIR / "src"

class ScriptRunRequest(BaseModel):
    script: str
    args: str

@app.get("/")
def read_root():
    return {"message": "RTLSTM Dashboard API"}

@app.get("/api/configs")
def list_configs():
    """List all JSON config files in inputs/"""
    if not INPUTS_DIR.exists():
        return []
    files = [f.name for f in INPUTS_DIR.glob("*.json")]
    return files

@app.get("/api/configs/{filename}")
def get_config(filename: str):
    """Get the content of a specific config file"""
    file_path = INPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configs/{filename}")
def save_config(filename: str, config: Dict[str, Any]):
    """Save content to a config file"""
    if not INPUTS_DIR.exists():
        INPUTS_DIR.mkdir()
        
    file_path = INPUTS_DIR / filename
    try:
        with open(file_path, "w") as f:
            json.dump(config, f, indent=4)
        return {"status": "success", "filename": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scripts")
def list_scripts():
    """List all Python scripts in src/"""
    if not SRC_DIR.exists():
        return []
    files = [f.name for f in SRC_DIR.glob("*.py")]
    return files

@app.post("/api/run")
def run_script(request: ScriptRunRequest):
    """Run a specific python script with arguments"""
    script_path = SRC_DIR / request.script
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Script not found")
    
    # Construct command
    # Using python3 from the environment where this runs
    # Security Note: This allows arbitrary execution of scripts in src/. 
    # Assumed safe for this local user tool.
    
    cmd = ["python3", str(script_path)] + request.args.split()
    
    try:
        # Run in the base directory so imports work if they expect CWD to be root
        result = subprocess.run(
            cmd, 
            cwd=BASE_DIR, 
            capture_output=True, 
            text=True
        )
        
        return {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
