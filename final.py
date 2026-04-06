import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional


def run_pf_afn_warp(
    person_id: str = "person",
    cloth_id: str = "cloth",
    project_root: Optional[str] = None,
    gpu_id: int = 0,
) -> Tuple[bool, str]:
    """
    Run PF-AFN warping model to generate warped cloth and mask.
    
    Args:
        person_id: Name of person image (without extension)
        cloth_id: Name of cloth image (without extension)
        project_root: Absolute path to DCI-VTON-Virtual-Try-On root directory.
                     If None, uses current working directory.
        gpu_id: GPU device ID (use -1 for CPU, though not recommended)
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    
    if project_root is None:
        project_root = os.getcwd()
    project_root = Path(project_root).resolve()
    
    print(f"\n{'='*70}")
    print(f"PF-AFN WARPING PIPELINE")
    print(f"{'='*70}")
    print(f"Project Root : {project_root}")
    print(f"Person ID    : {person_id}")
    print(f"Cloth ID     : {cloth_id}")
    print(f"GPU ID       : {gpu_id}")
    print(f"{'='*70}\n")
    
    # ====================================================================
    # STEP 1: Define and validate paths
    # ====================================================================
    print("[STEP 1/10] Validating PF-AFN setup...")
    
    dci_vton_test_dir = project_root / "datasets" / "custom" / "test"
    pf_afn_root = project_root / "PF-AFN" / "PF-AFN_test"
    pf_afn_data_dir = pf_afn_root / "data" / "test"
    pf_afn_results_dir = pf_afn_root / "results"
    test_py_path = pf_afn_root / "test.py"
    warp_checkpoint = pf_afn_root / "checkpoints" / "checkpoints" / "PFAFN" / "warp_model_final.pth"
    gen_checkpoint  = pf_afn_root / "checkpoints" / "checkpoints" / "PFAFN" / "gen_model_final.pth"
    
    if not pf_afn_root.exists():
        return False, f"PF-AFN directory not found: {pf_afn_root}"
    
    if not test_py_path.exists():
        return False, f"test.py not found: {test_py_path}"
    
    if not warp_checkpoint.exists():
        return False, f"Warp checkpoint not found: {warp_checkpoint}"
    
    if not gen_checkpoint.exists():
        return False, f"Gen checkpoint not found: {gen_checkpoint}"
    
    print(f"  ✓ PF-AFN root     : {pf_afn_root}")
    print(f"  ✓ test.py         : {test_py_path}")
    print(f"  ✓ Warp checkpoint : {warp_checkpoint}")
    print(f"  ✓ Gen checkpoint  : {gen_checkpoint}\n")
    
    # ====================================================================
    # STEP 2: Validate source files
    # ====================================================================
    print("[STEP 2/10] Validating source files...")
    
    source_mapping = {
        "image": (dci_vton_test_dir / "image", f"{person_id}.jpg"),
        "clothes": (dci_vton_test_dir / "cloth", f"{cloth_id}.jpg"),
        "edge": (dci_vton_test_dir / "cloth-mask", f"{cloth_id}.jpg"),
    }
    
    for folder_name, (source_dir, filename) in source_mapping.items():
        source_path = source_dir / filename
        if not source_path.exists():
            return False, f"Required file not found: {source_path}"
        file_size = source_path.stat().st_size / 1024
        print(f"  ✓ {folder_name:10s} : {filename:30s} ({file_size:.1f} KB)")
    print()
    
    # ====================================================================
    # STEP 3: Create PF-AFN data directory structure
    # ====================================================================
    print("[STEP 3/10] Creating PF-AFN data directories...")
    
    target_dirs = {}
    for folder_name in source_mapping.keys():
        target_dir = pf_afn_data_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_dirs[folder_name] = target_dir
        print(f"  ✓ {target_dir}")
    print()
    
    # ====================================================================
    # STEP 4: Copy files to PF-AFN data directory
    # ====================================================================
    print("[STEP 4/10] Copying files to PF-AFN data directory...")
    
    for folder_name, (source_dir, filename) in source_mapping.items():
        source_path = source_dir / filename
        target_path = target_dirs[folder_name] / filename
        
        try:
            shutil.copy2(source_path, target_path)
            print(f"  ✓ {filename:30s} → {folder_name}/")
        except Exception as e:
            return False, f"Failed to copy {source_path}: {str(e)}"
    print()
    
    # ====================================================================
    # STEP 5: Create test_pairs.txt
    # ====================================================================
    print("[STEP 5/10] Creating test_pairs.txt...")
    
    test_pairs_path = pf_afn_data_dir / "test_pairs.txt"
    pairs_content = f"{person_id}.jpg {cloth_id}.jpg"
    
    try:
        with open(test_pairs_path, "w", encoding="utf-8") as f:
            f.write(pairs_content + "\n")
        print(f"  ✓ Created : {test_pairs_path}")
        print(f"  ✓ Content : {pairs_content}\n")
    except Exception as e:
        return False, f"Failed to create test_pairs.txt: {str(e)}"
    
    # ====================================================================
    # STEP 6: Clean previous results
    # ====================================================================
    print("[STEP 6/10] Cleaning previous results...")
    
    if pf_afn_results_dir.exists():
        try:
            shutil.rmtree(pf_afn_results_dir)
            print(f"  ✓ Deleted old results: {pf_afn_results_dir}")
        except Exception as e:
            print(f"  ⚠ Warning: Could not delete results directory: {str(e)}")
    else:
        print(f"  ✓ No previous results to clean")
    print()
    
    # ====================================================================
    # STEP 7: Run PF-AFN test.py
    # ====================================================================
    print("[STEP 7/10] Running PF-AFN warping model...")
    
    cmd = [
        sys.executable,
        str(test_py_path),
        "--name", "demo",
        "--resize_or_crop", "None",
        "--batchSize", "1",
        "--gpu_ids", str(gpu_id),
    ]
    
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working directory: {pf_afn_root}")
    print(f"  {'─'*66}")
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(pf_afn_root)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(pf_afn_root),
        )
        
        for line in process.stdout:
            print(f"  [PF-AFN] {line.rstrip()}")
        
        return_code = process.wait()
        
        os.chdir(original_cwd)
        
        if return_code != 0:
            return False, f"PF-AFN test.py failed with exit code {return_code}"
        
        print(f"  {'─'*66}")
        print(f"  ✓ PF-AFN warping completed successfully\n")
        
    except Exception as e:
        os.chdir(original_cwd)
        return False, f"Failed to run PF-AFN test.py: {str(e)}"
    
    # ====================================================================
    # STEP 8: Verify generated outputs
    # ====================================================================
    print("[STEP 8/10] Verifying generated outputs...")
    
    pfafn_output_dir = pf_afn_results_dir / "demo" / "PFAFN"
    
    if not pfafn_output_dir.exists():
        return False, f"PF-AFN output directory not created: {pfafn_output_dir}"
    
    output_files = list(pfafn_output_dir.glob("*.jpg"))
    
    if not output_files:
        return False, f"No warped output found in: {pfafn_output_dir}"
    
    warped_output = output_files[0]
    file_size = warped_output.stat().st_size / 1024
    
    print(f"  ✓ Found warped cloth: {warped_output.name} ({file_size:.1f} KB)")
    print(f"  ✓ Location: {warped_output}\n")
    
    # ====================================================================
    # STEP 9: Copy outputs back to DCI-VTON
    # ====================================================================
    print("[STEP 9/10] Copying outputs back to DCI-VTON...")
    
    dci_warp_dir = dci_vton_test_dir / "cloth-warp"
    dci_warp_mask_dir = dci_vton_test_dir / "cloth-warp-mask"
    
    dci_warp_dir.mkdir(parents=True, exist_ok=True)
    dci_warp_mask_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        dci_warp_output = dci_warp_dir / f"{cloth_id}.png"
        shutil.copy2(warped_output, dci_warp_output)
        print(f"  ✓ Warped cloth → {dci_warp_output}")
        
        dci_warp_mask_output = dci_warp_mask_dir / f"{cloth_id}.png"
        original_mask = dci_vton_test_dir / "cloth-mask" / f"{cloth_id}.jpg"
        
        if original_mask.exists():
            shutil.copy2(original_mask, dci_warp_mask_output)
            print(f"  ✓ Warp mask   → {dci_warp_mask_output}")
        else:
            print(f"  ⚠ Warning: Original mask not found, skipping mask copy")
        
    except Exception as e:
        return False, f"Failed to copy outputs back to DCI-VTON: {str(e)}"
    
    print()
    
    # ====================================================================
    # STEP 10: Final cleanup and summary
    # ====================================================================
    print("[STEP 10/10] Pipeline summary...")
    print(f"  ✓ Warped cloth ready : {dci_warp_dir}")
    print(f"  ✓ Warp mask ready    : {dci_warp_mask_dir}")
    print()
    
    print(f"{'='*70}")
    print(f"✓ PF-AFN WARPING PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")
    
    return True, "Warping completed successfully"


if __name__ == "__main__":
    success, message = run_pf_afn_warp(
        person_id="person",
        cloth_id="cloth",
        gpu_id=0
    )
    
    if success:
        print(f"\n✓ SUCCESS: {message}")
        sys.exit(0)
    else:
        print(f"\n✗ ERROR: {message}")
        sys.exit(1)