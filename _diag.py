import json
import sys
sys.path.insert(0, 'Pipeline')

# Check segment dataset
with open('segment/train/_annotations.coco.json') as f:
    data = json.load(f)

print("=== SEGMENT ===")
for c in data['categories']:
    print(f"  Cat: {c}")
print(f"  Num images: {len(data['images'])}")
print(f"  Num anns: {len(data['annotations'])}")
ann = data['annotations'][0]
for k, v in ann.items():
    if k == 'segmentation':
        print(f"  {k}: type={type(v).__name__}, len={len(v)}")
    else:
        print(f"  {k}: {v}")
print(f"  Image 0: {data['images'][0]}")

# Check minicoco
with open('minicoco/train/_annotations.coco.json') as f:
    data2 = json.load(f)
print("\n=== MINICOCO ===")
for c in data2['categories'][:5]:
    print(f"  Cat: {c}")
print(f"  Total cats: {len(data2['categories'])}")
print(f"  Num images: {len(data2['images'])}")
print(f"  Num anns: {len(data2['annotations'])}")
ann2 = data2['annotations'][0]
for k, v in ann2.items():
    if k == 'segmentation':
        print(f"  {k}: type={type(v).__name__}, len={len(v)}")
    else:
        print(f"  {k}: {v}")
print(f"  Image 0: {data2['images'][0]}")

# Check imagenet
with open('imagenet/imagenet1000_clsidx_to_labels.json') as f:
    labels = json.load(f)
print("\n=== IMAGENET ===")
print(f"  Total classes: {len(labels)}")
for k in list(labels.keys())[:5]:
    print(f"  {k}: {labels[k]}")
