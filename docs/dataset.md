# Dataset Layout

The training scripts expect a YOLO-style dataset prepared locally:

```text
YOLO_Master_Dataset/
|-- data.yaml
|-- images/
|   |-- train/
|   `-- val/
`-- labels/
    |-- train/
    `-- val/
```

Example class mapping:

```yaml
names:
  0: bike
  1: fallen_bike
  2: parking_area
```

Datasets are intentionally not included in the public repository. Keep raw images, labels, and any media that may contain privacy-sensitive content outside Git.
