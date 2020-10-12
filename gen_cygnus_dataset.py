import numpy as np
import bpy
import starfish
import starfish.annotation
from mathutils import Euler
import sys
import json
import time
import os
import boto3
import shortuuid
import yaml
import subprocess
import tqdm
"""
    script for generating cygnus training data with glare, blur, and domain randomized backgrounds.
"""


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    return activated_gpus


enable_gpus("CUDA", True)
sys.stdout = sys.stderr

BACKGROUND_COLOR = (0, 0, 0)
LABEL_MAP_FULL = {
    'barrel': (206, 0, 0),
    'panel_right': (206, 206, 0),
    'panel_left': (0, 0, 206),
    'orbitrak_logo': (0, 206, 206),
    'cygnus_logo': (206, 0, 206)
}
LABEL_MAP_SINGLE = {'cygnus': list(LABEL_MAP_FULL.values())}

OG_KEYPOINTS = {
    'barrel_center': (0.0, 0.0, 0.0),
    'panel_left': (0.0, -3.3931, -3.4995),
    'panel_right': (0.0, 3.3931, -3.4995),
    'cygnus_logo': (1.524, 0.0, 0.8718),
    'orbitrak_logo': (1.524, 0.0, 0.2359),
    'barrel_bottom': (0, 0, -3.6295),
    'barrel_top': (0, 0, 3.18566)
}

GLARE_TYPES = ['FOG_GLOW', 'SIMPLE_STAR', 'STREAKS', 'GHOSTS']


def check_nodes(filters, node_tree):
    """
        check if requested filters are in node tree of given blender file
    """
    _filters = []
    for f in filters:
        if f in node_tree.nodes.keys():
            _filters.append(f)
        else:
            print("{} is not in the node tree".format(f))
            sys.exit()
    return _filters


def reset_filter_nodes(node_tree):
    """
        resets filters nodes to default values that will not modify final image
    """
    
    if 'Glare' in node_tree.nodes.keys():
        node_tree.nodes['Glare'].mix = -1
        node_tree.nodes['Glare'].threshold = 8
    
    if 'Blur' in node_tree.nodes.keys():
        node_tree.nodes['Blur'].size_x = 0
        node_tree.nodes['Blur'].size_y = 0


def set_filter_nodes(filters, node_tree):
    """
        set filter node parameters to random value
    """
    result_dict = {
        'Glare':{
            'mix':-1,
            'threshold': 8,
            'type': 'None'
        },
        
        'Blur':{
            'size_x':0,
            'size_y':0
        }
    }
    if 'Glare' in filters:
        
        glare_value = 0.5
        glare_type = np.random.randint(0,4)
        glare_threshold = np.random.beta(2,8)
        # configure glare node
        node_tree.nodes["Glare"].glare_type = result_dict['Glare']['type'] = GLARE_TYPES[glare_type]
        node_tree.nodes["Glare"].mix = result_dict['Glare']['mix'] = glare_value
        node_tree.nodes["Glare"].threshold = result_dict['Glare']['threshold'] = glare_threshold

    if 'Blur' in filters:
        # set blur values
        blur_x = np.random.uniform(10, 30)
        blur_y = np.random.uniform(10, 30)
        node_tree.nodes["Blur"].size_x = result_dict['Blur']['size_x'] = blur_x
        node_tree.nodes["Blur"].size_y = result_dict['Blur']['size_y'] = blur_y
    return result_dict


def get_occluded_offsets(num):
    offsets =[]
    while len(offsets) < num:
        x_y = np.random.uniform(-.05, 1.05, size=(2,))
        if not ( 0.1 < x_y[0] < 0.9 and 0.1 < x_y[1] < 0.9 ):
            offsets.append(x_y)
    return offsets


# render resolution
RES_X = 1024
RES_Y = 1024


def generate(ds_name,
             num,
             filters,
             occlusion=None,
             bucket=None,
             background_dir=None):
    start_time = time.time()

    # check if folder exists in render, if not, create folder
    try:
        os.mkdir(os.path.join("render", ds_name))
    except Exception:
        pass
    
    tags = "cygnus " + str(num)
    for f in filters:
        tags += ' ' + f  

    data_storage_path = os.path.join(os.getcwd(), "render", ds_name)

    enable_gpus("CUDA", True)
    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path

    # remove all animation
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            obj.animation_data_clear()
    bpy.context.scene.frame_set(0)

    shortuuid.set_alphabet('12345678abcdefghijklmnopqrstwxyz')
    if occlusion:
        offsets = get_occluded_offsets(num)
        tags += ' occlusion'
    else:
        offsets = np.random.uniform(low=0.15, high=.85, size=(num,2))
    sequence = starfish.Sequence.standard(
        pose=starfish.utils.random_rotations(num),
        lighting=starfish.utils.random_rotations(num),
        background=starfish.utils.random_rotations(num),
        distance=np.random.uniform(low=35, high=75, size=(num,)),
        offset=offsets
    )

    keypoints = starfish.annotation.generate_keypoints(bpy.data.objects['Cygnus_Real'], 128, seed=4)

    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()

    metadata = {
        'keypoints': keypoints,
        'og_keypoints': OG_KEYPOINTS,
        'label_map': LABEL_MAP_FULL
    }
    with open(os.path.join(data_storage_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    with open(os.path.join(data_storage_path, 'gen_code.py'), 'w') as f:
        f.write(code)
    
    num_images = 0
    bpy.data.scenes['Render'].render.resolution_x = RES_X
    bpy.data.scenes['Render'].render.resolution_y = RES_Y
    # get images from background directory
    if background_dir is not None:
        images_list = []
        for f in os.listdir(background_dir):
            if f.endswith(".exr") or f.endswith(".jpg") or f.endswith(".png"):
                images_list.append(f)
        images_list = sorted(images_list)
        num_images = len(images_list)
        if num_images > 0:
            tags += ' randomized backgrounds'

    node_tree = bpy.data.scenes["Render"].node_tree
    reset_filter_nodes(node_tree)
    
    # set default background in case base blender file is messed up
    bpy.data.worlds["World"].node_tree.nodes['Environment Texture'].image = bpy.data.images["HDRI_Earth_3.exr"]
    
    # set background image mode depending on nodes in tree either sets environment texture or image node
    # NOTE: if using image node it is recommended that you add a crop node to perform random crop on images.
    # WARNING: this only looks to see if nodes are in the node tree. does not check if they are connected properly.
    image_node_in_tree = 'Image' in bpy.data.scenes['Render'].node_tree.nodes.keys()
    if image_node_in_tree:
        random_crop = 'Crop' in bpy.data.scenes['Render'].node_tree.nodes.keys()

    for i, frame in enumerate(tqdm.tqdm(sequence)):
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Cygnus_Real"], bpy.data.objects["Camera_Real"], bpy.data.objects["Sun"])
        frame.setup(bpy.data.scenes['Mask_ID'], bpy.data.objects["Cygnus_MaskID"], bpy.data.objects["Camera_MaskID"], bpy.data.objects["Sun"])

        # create name for the current image (unique to that image)
        name = shortuuid.uuid()
        output_node.file_slots[0].path = "image_#" + str(name)
        output_node.file_slots[1].path = "mask_#" + str(name)

        # set background image, using image node and crop node if in tree, otherwise just set environment texture.
        if num_images > 0:
            background_image = np.random.choice(images_list)
            image = bpy.data.images.load(filepath = os.getcwd()+ '/' + background_dir + '/' + background_image)
            frame.background = str(background_image)
            if image_node_in_tree:
                if random_crop: 
                    if RES_X < image.size[0]:
                        frame.crop_x = off_x = np.random.randint(0, image.size[0]-RES_X-1)
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].min_x = off_x
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].max_x = off_x + RES_X
                    else:
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].min_x = 0
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].max_x = image.size[0]
                    if RES_Y < image.size[1]:
                        frame.crop_y = off_y = np.random.randint(0, image.size[1]-RES_Y-1)
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].min_y = off_y
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].max_y = off_y + RES_Y
                    else:
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].min_y = 0
                        bpy.data.scenes["Render"].node_tree.nodes["Crop"].max_y = image.size[1]
                bpy.data.scenes['Render'].node_tree.nodes['Image'].image = image
            else:
                bpy.data.worlds["World"].node_tree.nodes['Environment Texture'].image = image

        # set filters to random values
        frame.augmentations = set_filter_nodes(filters, node_tree)
        
        # render
        bpy.ops.render.render(scene="Render")
        # mask/bbox stuff
        mask = starfish.annotation.normalize_mask_colors(os.path.join(data_storage_path, f'mask_0{name}.png'),
                                                         list(LABEL_MAP_SINGLE.values())[0] + [BACKGROUND_COLOR])
        frame.bboxes = starfish.annotation.get_bounding_boxes_from_mask(mask, LABEL_MAP_SINGLE)
        frame.centroids = starfish.annotation.get_centroids_from_mask(mask, LABEL_MAP_SINGLE)
        frame.keypoints = starfish.annotation.project_keypoints_onto_image(keypoints, bpy.data.scenes['Real'],
                                                                           bpy.data.objects['Cygnus_Real'], bpy.data.objects['Camera_Real'])
        og_keypoints = starfish.annotation.project_keypoints_onto_image(OG_KEYPOINTS.values(), bpy.data.scenes['Real'],
                                                                        bpy.data.objects['Cygnus_Real'], bpy.data.objects['Camera_Real'])
        frame.og_keypoints = {k: v for k, v in zip(OG_KEYPOINTS.keys(), og_keypoints)}

        frame.sequence_name = ds_name
        frame.tags = tags
        frame.focal_length = bpy.data.cameras["Camera"].lens
        frame.sensor_width = bpy.data.cameras["Camera"].sensor_width
        frame.sensor_height = bpy.data.cameras["Camera"].sensor_height
        frame.lens_unit = bpy.data.cameras["Camera"].lens_unit

        # dump data to json
        with open(os.path.join(output_node.base_path, "meta_0" + str(name)) + ".json", "w") as f:
            f.write(frame.dumps())
            f.write('\n')
    if bucket:
        upload(ds_name, bucket)
    print("===========================================" + "\r")
    time_taken = time.time() - start_time
    print("------Time Taken: %s seconds----------" % (time_taken) + "\r")
    print("Number of images generated: " + str(i) + "\r")
    print("Average time per image: " + str(time_taken / i))
    print("Data stored at: " + data_storage_path)
    bpy.ops.wm.quit_blender()
    
def upload(ds_name, bucket_name):
    print("\n\n______________STARTING UPLOAD_________")

    subprocess.run(['aws', 's3', 'sync', os.path.join('render', ds_name), f's3://{bucket_name}/{ds_name}'])
    time.sleep(5)
    # delete local imageset to save space on lab computer.
    subprocess.run(['rm', '-rf', os.path.join('render', ds_name)])

def validate_bucket_name(bucket_name):
    s3t = boto3.resource('s3')
    # check if bucket exits. If not return false
    if s3t.Bucket(bucket_name).creation_date is None:
        print("...Bucket does not exist, enter valid bucket name...")
        return False
    else:
        # if exists, return true
        print("...bucket exists....")
        return True


def main():
    try:
        os.mkdir("render")
    except Exception:
        pass

    config_path = input("*> Enter path to config.yaml file: ")
    while not os.path.isfile(config_path):
        config_path = input("*> Enter path to config.yaml file: ")
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    bucket = config.get("s3_bucket")
    if bucket:
        while not validate_bucket_name(bucket):
            bucket = input("*> Enter Bucket name: ")

    imagesets = config.get("imagesets")
    if imagesets:
        imgset_dict = {imgset: {
            'filters': imagesets[imgset].get('filters', []),
            'num': min(int(imagesets[imgset].get('num', 10)), 10000),
            'occlusion': imagesets[imgset].get('occlusion', False),
            'backgrounds': imagesets[imgset].get('backgrounds'),
            }
            for imgset in imagesets.keys()}
        print(imgset_dict)
        node_tree = bpy.data.scenes["Render"].node_tree
        
        for imgset in imgset_dict.keys():
            set_conf = imgset_dict[imgset]
            background_dir = set_conf['backgrounds']
            if background_dir:
                if not os.path.isdir(background_dir):
                    print(f'Randomized background dir for {imgset} does not exist')
                    sys.exit()
            if len(set_conf['filters']) > 0:
                imgset_dict[imgset]['filters'] = check_nodes([f.title() for f in set_conf['filters']], node_tree)
        
        for imgset in imgset_dict.keys():
            set_conf = imgset_dict[imgset]
            generate(imgset, set_conf['num'], set_conf['filters'], set_conf['occlusion'], bucket, set_conf['backgrounds'])
    print("______________DONE EXECUTING______________")


if __name__ == "__main__":
    main()
