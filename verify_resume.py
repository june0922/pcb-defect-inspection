import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    print("Starting verification: Continuous vs Resume Training...")
    
    # Paths
    project_root = Path(".").resolve()
    processed_dir = project_root / "preprocessed_data"
    test_runs_dir = project_root / "runs" / "verify_resume_3"
    
    if test_runs_dir.exists():
        shutil.rmtree(test_runs_dir)
    test_runs_dir.mkdir(parents=True, exist_ok=True)
    
    # We will use the existing preprocessed_data train/val files.
    # To make it fast, let's use a very small subset or just small epochs.
    # We'll use 4 epochs for the total run.
    total_epochs = 2
    mid_epochs = 1
    
    import yaml
    
    # Create a tiny dataset to make training fast
    tiny_dir = test_runs_dir / "tiny_dataset"
    for split in ["train", "val"]:
        (tiny_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (tiny_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        imgs = list((processed_dir / "images" / split).glob("*.jpg"))[:16] # only 16 images
        for img in imgs:
            shutil.copy(img, tiny_dir / "images" / split / img.name)
            lbl = processed_dir / "labels" / split / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy(lbl, tiny_dir / "labels" / split / lbl.name)
                
    with open(project_root / "data.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = str(tiny_dir.resolve())
    data_yaml = test_runs_dir / "data_test.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f)
    
    # 1. Continuous Training
    print("\n--- 1. Continuous Training (2 epochs) ---")
    model_cont = YOLO("weights/yolo26s.pt")
    results_cont = model_cont.train(
        data=str(data_yaml),
        epochs=total_epochs,
        project=str(test_runs_dir),
        name="continuous",
        imgsz=640, # smaller for speed
        batch=32,
        device=0 if torch.cuda.is_available() else 'cpu',
        cache="ram",
        verbose=False
    )
    
    # 2. Interrupted Training
    print(f"\n--- 2. Interrupted Training (first {mid_epochs} epochs) ---")
    model_int = YOLO("weights/yolo26s.pt")
    model_int.train(
        data=str(data_yaml),
        epochs=mid_epochs,
        project=str(test_runs_dir),
        name="resume_test",
        imgsz=640,
        batch=32,
        device=0 if torch.cuda.is_available() else 'cpu',
        cache="ram",
        verbose=False
    )
    
    # 3. Resumed Training
    print(f"\n--- 3. Resumed Training (remaining epochs up to {total_epochs}) ---")
    last_pt = test_runs_dir / "resume_test" / "weights" / "last.pt"
    model_res = YOLO(str(last_pt))
    results_res = model_res.train(
        data=str(data_yaml),
        epochs=total_epochs, # when resuming, it expects the total epochs to reach
        project=str(test_runs_dir),
        name="resume_test", # same name, will append to it conceptually, or we can use resume=True
        resume=True,
        imgsz=640,
        batch=32,
        device=0 if torch.cuda.is_available() else 'cpu',
        cache="ram",
        verbose=False
    )
    
    # Compare
    print("\n--- Comparison ---")
    cont_map = results_cont.box.map
    res_map = results_res.box.map
    
    print(f"Continuous mAP50-95: {cont_map:.4f}")
    print(f"Resumed mAP50-95:    {res_map:.4f}")
    
    if abs(cont_map - res_map) < 1e-4:
        print("=> SUCCESS: Results are identical (within float margin). Resume works perfectly.")
    else:
        print("=> WARNING: Results differ. There might be some state not fully restored (e.g. dataloader shuffle, random seed, optimizer state).")

if __name__ == "__main__":
    import torch
    main()
