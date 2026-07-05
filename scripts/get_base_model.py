import yaml
from pathlib import Path
import sys

def main():
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print("weights/yolov8n.pt") # fallback if config not found
        sys.exit(0)
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Default to weights/yolov8n.pt if not found in config
            model_path = config.get("train", {}).get("model", "weights/yolov8n.pt")
            print(model_path)
    except Exception as e:
        print("weights/yolov8n.pt")
        sys.exit(0)

if __name__ == "__main__":
    main()
