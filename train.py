from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")

    results = model.train(
        data="data/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        name="car_damage_model_v2",
        patience=20,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        freeze=10
    )

    print("Training complete!")
    print("Best model:", results.save_dir)

if __name__ == "__main__":
    main()