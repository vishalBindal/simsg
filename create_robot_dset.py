import pickle
import torch
import numpy as np
import shutil
import os
import torch
import cv2
from torchvision.ops.boxes import box_iou
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

'''
This script can be used to convert dataset to the following form
- a folder containing scene images (labeled as 0.png, 1.png and so on)
- a file instances.json containing COCO-format annotations (structure below)
'''

## Config ##
dataset_dir = '../input/robotmanipulation/complete_dataset'
mode = 'val' # dataset_dir/mode+'_images' will be looked up
scene_no = 0 # 0==input scene, 1==final_scene
get_masks = False # whether to get mask in annotations
###############


'''
Annotations needed:

instances_json
{
'images': [{
    'id':,
    'file_name':,
    'width':,
    'height':
}],
'categories': [{
    'id':,
    'name':
}],
'annotations': [{
    'image_id':,
    'bbox':,
    'category_id':,
    'mask':
}]
}
'''

images_dir = mode + '_images'
if os.path.isdir(images_dir):
    shutil.rmtree(images_dir)
os.mkdir(images_dir)
    
instances_file = 'instances_' + mode +'.json'

def load_pickle(file_name):
    f = open(file_name, 'rb')
    data = pickle.load(f)
    f.close()
    return data

data = load_pickle(os.path.join(dataset_dir, mode+'.pkl'))

images = []
categories = []
annotations = []

colors = ['blue','green','red','cyan','yellow','magenta','white']
types = ['small', 'lego']
category_id = 0
for typ in types:
    for color in colors:
        categories.append({
            'id': category_id,
            'name': typ + '_' + color
        })
        category_id += 1
        
for i, scene in enumerate(sorted(os.listdir(os.path.join(dataset_dir, mode)))):
    if i%20==0:
        print(i, scene)
    image_path = os.path.join(dataset_dir, mode, scene, 'S00', 'rgba.png')
    h,w,_ = cv2.imread(image_path).shape
    shutil.copyfile(image_path, os.path.join(images_dir, f'{i}.png'))
    
    images.append({
        'id':i,
        'file_name':f'{i}.png',
        'width':w,
        'height':h
    })
    
    if get_masks:
        mask_path = os.path.join(dataset_dir, mode, scene, 'S00', 'mask.png')
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        obj_ids = np.unique(mask)
        obj_masks = []
        for o_id in obj_ids:
            if o_id==0:
                continue
            unit_mask = np.where(mask == o_id, 1, 0).astype(np.int8)
            
            py, px = np.where(mask == o_id)
            bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
            
            unit_mask = unit_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            bbox = torch.Tensor([bbox])
            
            obj_masks.append((bbox, unit_mask))
            
        masks_chosen = set()

    for j in range(6):
        bbox = data['objects'][i][0][j][:4].tolist()
        x1,y1,x2,y2 = h * bbox[0], w * bbox[1], h * bbox[2], w * bbox[3]
        bbox = [y1,x1,y2-y1,x2-x1]
        category = data['object_color'][i][j]
        if data['object_type'][i][j] != 'small':
            category = category + 7
        
        if get_masks:
            max_iou = 0
            best_index = 0
            for k, (bb, _) in enumerate(obj_masks):
                iou = box_iou(bb, torch.Tensor([[y1,x1,y2,x2]]))
                if iou > max_iou:
                    best_index = k
                    max_iou = iou
            
            # sanity check
            assert best_index not in masks_chosen
            masks_chosen.add(best_index)
        
        
            annotations.append({
                'image_id': i,
                'bbox': bbox,
                'category_id': category,
                'mask': obj_masks[best_index][1].tolist()
            })
        
        else:
            annotations.append({
                'image_id': i,
                'bbox': bbox,
                'category_id': category
            })

        
instances = {
    'images': images,
    'categories': categories,
    'annotations': annotations
}


import json
with open(instances_file, "w") as f:
    json.dump(instances, f)



def show_template(img, bb):
    fig, ax = plt.subplots()
    ax.imshow(img[:,:,::-1])
    rect = patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()

# mask = cv2.imread('../input/robotmanipulation/complete_dataset/val/0007/S00/mask.png')
# mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# obj_ids = np.unique(mask)

# py, px = np.where(mask == 3)
# bbox = torch.Tensor([np.min(px), np.min(py), np.max(px), np.max(py)])

# bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        
# img = cv2.imread('../input/robotmanipulation/complete_dataset/val/0007/S00/rgba.png')
# # h,w = img.shape[:2]
# # bbox = data['objects'][0][0][0][:4].tolist()
# # x1,y1,x2,y2 = h * bbox[0], w * bbox[1], h * bbox[2], w * bbox[3]
# # bbox = [y1,x1,y2-y1,x2-x1]
# print(bbox)
# show_template(img, bbox)