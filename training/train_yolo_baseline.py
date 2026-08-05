import argparse
import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_name = args.model if args.model else config['model']['backbone'] + '.pt'
    if args.resume:
        model = YOLO(config['paths']['baseline_weights'])
    else:
        model = YOLO(model_name)
    
    epochs = args.epochs if args.epochs else config['training']['epochs_baseline']
    lr = args.lr if args.lr else 0.005
    run_name = args.name if args.name else 'baseline'
    
    # Auto-clean any stale Ultralytics dataset cache files (*.cache)
    if sys.platform == 'win32':
        os.system('del /f /q /s data\\*.cache >nul 2>&1')
    for cache_file in Path("data").rglob("*.cache"):
        try:
            os.remove(cache_file)
            print(f"Removed stale cache: {cache_file}")
        except Exception:
            pass

    data_yaml_path = args.data if args.data else config['paths']['dataset_yaml']

    # Dynamic resolution of absolute dataset root directory
    yaml_abs = Path(data_yaml_path).resolve()
    yaml_name_lower = str(yaml_abs).lower()
    if (yaml_abs.parent / "train").exists():
        root_dir = yaml_abs.parent
    elif "wood" in yaml_name_lower and "3k" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "wood_3k").resolve()
    elif "aluminum" in yaml_name_lower and "3k" in yaml_name_lower:
        root_dir = (Path("data") / "processed_aluminum_3k").resolve()
    elif "wood" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "wood_10class").resolve()
    elif "aluminum" in yaml_name_lower:
        root_dir = (Path("data") / "processed_aluminum").resolve()
    elif "steel" in yaml_name_lower:
        root_dir = (Path("data") / "processed" / "steel_unified").resolve()
    else:
        root_dir = yaml_abs.parent

    with open(data_yaml_path, 'r') as f:
        ds_yaml = yaml.safe_load(f)
    
    ds_yaml['path'] = str(root_dir).replace('\\', '/')
    with open(data_yaml_path, 'w') as f:
        yaml.dump(ds_yaml, f, sort_keys=False)

    # On Windows, workers > 0 causes multiprocessing overhead — use 0
    num_workers = 0 if sys.platform == 'win32' else 4

    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=args.imgsz if args.imgsz else config['training']['image_size'],
        batch=args.batch if args.batch else config['training']['batch_size'],
        device=args.device,
        optimizer=config['training']['optimizer'],
        lr0=float(lr),
        lrf=0.01,
        weight_decay=float(config['training']['weight_decay']),
        cos_lr=True,
        patience=30,
        amp=True,          # FP16 mixed precision: ~1.5-2x faster, half VRAM usage
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        flipud=0.5,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        project='runs',
        name=run_name,
        workers=num_workers,
        cache='ram',       # Load all images into RAM — critical for HDD users (3-4x faster per epoch)
        exist_ok=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline YOLOv10 model.")
    parser.add_argument('--config', type=str, default='training/config.yaml', help='Path to config file.')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset.yaml.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--model', type=str, default=None, help='YOLO model variant (e.g. yolov10m.pt, yolov10l.pt).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda).')
    parser.add_argument('--batch', type=int, default=None, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size.')
    parser.add_argument('--name', type=str, default=None, help='Run name (e.g. steel, aluminum, wood).')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate override.')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint.')
    args = parser.parse_args()
    main(args)
