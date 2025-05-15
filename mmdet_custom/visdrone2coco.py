import os
import json
import cv2
from tqdm import tqdm

def visdrone2coco(visdrone_img_dir, visdrone_ann_dir, output_json):
    # Complete VisDrone category mapping (10 valid classes)
    categories = [
        {"id": 1, "name": "pedestrian"},
        {"id": 2, "name": "person"},
        {"id": 3, "name": "bicycle"},
        {"id": 4, "name": "car"},
        {"id": 5, "name": "van"},
        {"id": 6, "name": "truck"},
        {"id": 7, "name": "tricycle"},
        {"id": 8, "name": "awning-tricycle"},
        {"id": 9, "name": "bus"},
        {"id": 10, "name": "motor"}
    ]
    
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    ann_id = 1

    dir_contents = sorted(os.listdir(visdrone_img_dir))
    dir_len = len(dir_contents)

    factor = 1
    dir_contents = dir_contents[:dir_len // factor]
    print(f"Processing {dir_len // factor} images...")  

    for img_file in tqdm(sorted(os.listdir(visdrone_img_dir))):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(visdrone_img_dir, img_file)
        ann_file = img_file.rsplit('.', 1)[0] + '.txt'
        ann_path = os.path.join(visdrone_ann_dir, ann_file)
        
        # Get image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        
        # Add image info
        image_id = len(coco_output["images"]) + 1
        coco_output["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })
        
        # Parse annotations if annotation file exists
        if not os.path.exists(ann_path):
            continue
            
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                    
                x, y, w, h = map(float, parts[:4])
                category_id = int(parts[5])
                
                # Skip ignored regions (category 0) and invalid categories
                if category_id < 1 or category_id > 10:
                    continue
                
                # Convert to COCO bbox format [x,y,width,height]
                coco_output["annotations"].append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "ignore": 0
                })
                ann_id += 1
    
    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=2)

if __name__ == "__main__":
    # Example usage for train set
    visdrone2coco(
        visdrone_img_dir="data/VisiDrone/train/images",
        visdrone_ann_dir="data/VisiDrone/train/annotations",
        output_json="data/VisiDrone/annotations/train_coco.json"
    )
  