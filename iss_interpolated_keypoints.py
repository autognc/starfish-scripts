import numpy as np
import bpy
import starfish
from starfish import Frame, Sequence
import starfish.annotation
from mathutils import Euler
import math
import sys
import json
import time
import os
import boto3
import shortuuid
import subprocess
import tqdm
"""
    script for generating cygnus training data with glare, blur, and domain randomized backgrounds.
"""


#TODO add support for Optix
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

LABEL_MAP_SINGLE = {'iss': [(255,255,255)]}


# Defaults and constants
# render resolution
RES_X = 1024
RES_Y = 1024
# exposure and background strength defaults for new iss model and hdri background
EXPOSURE_DEFAULT = -8.15 
BACKGROUND_STRENGTH_DEFAULT = 0.312
NUM = 10


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
    


def generate(ds_name, tags):
    start_time = time.time()

    # check if folder exists in render, if not, create folder
    try:
        os.mkdir(os.path.join("render", ds_name))
    except Exception:
        pass

    data_storage_path = os.path.join(os.getcwd(), "render", ds_name)

    enable_gpus("CUDA", True)
    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path

    # remove all animation
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            obj.animation_data_clear()
    bpy.context.scene.frame_set(0)

    # set color management
    for scene in bpy.data.scenes:
        scene.view_settings.view_transform = 'Filmic'
        scene.view_settings.look = 'High Contrast'
    shortuuid.set_alphabet('12345678abcdefghijklmnopqrstwxyz')
    def E(*args):
        return Euler([-math.pi, 0, 0]).to_quaternion() @ Euler(*args).to_quaternion()

    waypoints = [
        Frame(pose=E((math.radians(-45), math.radians(-35), math.radians(0)), 'XYZ'), distance=1000, offset=(0.35, 0.35), background=Euler((math.radians(0), math.radians(0), math.radians(0)), 'XYZ')),
        Frame(pose=E((math.radians(-30), math.radians(-40), math.radians(-10)), 'XYZ'), distance=1200, offset=(0.35, 0.5), background=Euler((math.radians(20), math.radians(0), math.radians(0)), 'XYZ')),
        Frame(pose=E((math.radians(0), math.radians(-50), math.radians(20)), 'XYZ'), distance=1100, offset=(0.5, 0.5), background=Euler((math.radians(44), math.radians(0), math.radians(-5)), 'XYZ')),
        Frame(pose=E((math.radians(30), math.radians(-50), math.radians(-10)), 'XYZ'), distance=800, offset=(0.55, 0.55), background=Euler((math.radians(70), math.radians(0), math.radians(-20)), 'XYZ')),
        Frame(pose=E((math.radians(60), math.radians(-55), math.radians(-10)), 'XYZ'), distance=600, offset=(0.55, 0.6), background=Euler((math.radians(90), math.radians(0), math.radians(-40)), 'XYZ')),
    ]

    counts = [120, 120, 120, 120]

    sequence = starfish.Sequence.interpolated(waypoints, counts)

    keypoints = starfish.annotation.generate_keypoints(bpy.data.objects['ISS_PIVOT'], 128, seed=6)

    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()

    metadata = {
        'keypoints': keypoints,
        'label_map': LABEL_MAP_SINGLE
    }

    with open(os.path.join(data_storage_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    with open(os.path.join(data_storage_path, 'gen_code.py'), 'w') as f:
        f.write(code)
    
    num_images = 0
    bpy.data.scenes['Render'].render.resolution_x = RES_X
    bpy.data.scenes['Render'].render.resolution_y = RES_Y

    node_tree = bpy.data.scenes["Render"].node_tree
    reset_filter_nodes(node_tree)
    
    for i, frame in enumerate(tqdm.tqdm(sequence)):
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["ISS_PIVOT"], bpy.data.objects["Camera_Real"], bpy.data.objects["Sun"])

        # create name for the current image (unique to that image)
        name = str(i).zfill(5)
        output_node.file_slots[0].path = "image_#" + str(name)
        output_node.file_slots[1].path = "mask_#" + str(name)

        # render
        bpy.ops.render.render(scene="Render")
        # mask/bbox stuff
        mask = starfish.annotation.normalize_mask_colors(os.path.join(data_storage_path, f'mask_0{name}.png'),
                                                         list(LABEL_MAP_SINGLE.values())[0] + [BACKGROUND_COLOR])
        frame.bboxes = starfish.annotation.get_bounding_boxes_from_mask(mask, LABEL_MAP_SINGLE)
        frame.centroids = starfish.annotation.get_centroids_from_mask(mask, LABEL_MAP_SINGLE)
        frame.keypoints = starfish.annotation.project_keypoints_onto_image(keypoints, bpy.data.scenes['Real'],
                                                                           bpy.data.objects['ISS_PIVOT'], bpy.data.objects['Camera_Real'])

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

    yes = {'y', 'Y', 'yes'}
    runUpload = input("*> Would you like to upload these images to AWS? [y/n]: ")
    if runUpload in yes:
        bucket_name = input("*> Enter Bucket name: ")
        # check if bucket name valid
        while not validate_bucket_name(bucket_name):
            bucket_name = input("*> Enter Bucket name: ")

    dataset_name = input("*> Enter name for dataset/folder: ")
    print("   Note: rendered images will be stored in a directory called 'render' in the same local directory this script is located under the directory name you specify.")
    tags = input("*> Enter tags for the batch seperated with space: ")

    tags_list = tags.split()
    
    generate(dataset_name, tags_list)
    if runUpload in yes:
        upload(dataset_name, bucket_name)
    print("______________DONE EXECUTING______________")


if __name__ == "__main__":
    main()
