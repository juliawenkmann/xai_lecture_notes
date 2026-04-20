# CUB Data Layout

Place the official `CUB_200_2011` download under:

```text
data/cub/raw/CUB_200_2011/
```

The notebook expects the standard raw files from the dataset release, including:

- `images.txt`
- `image_class_labels.txt`
- `train_test_split.txt`
- `classes.txt`
- `images/`
- `attributes/attributes.txt`
- `attributes/image_attribute_labels.txt`

The notebook will create cached image features under `data/cub/cache/`.
