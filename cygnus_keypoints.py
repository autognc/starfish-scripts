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
import subprocess
import tqdm

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
    bpy.context.scene.cycles.device = "GPU"

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
NUM = 10000


def generate(ds_name):
    start_time = time.time()

    # check if folder exists in render, if not, create folder
    try:
        os.mkdir(os.path.join("render", ds_name))
    except Exception:
        pass

    data_storage_path = os.path.join(os.getcwd(), "render", ds_name)

    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path

    # remove all animation
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            obj.animation_data_clear()
    bpy.context.scene.frame_set(0)

    shortuuid.set_alphabet('12345678abcdefghijklmnopqrstwxyz')

    sequence = starfish.Sequence.standard(
        pose=starfish.utils.random_rotations(NUM),
        lighting=starfish.utils.random_rotations(NUM),
        background=starfish.utils.random_rotations(NUM),
        distance=np.random.uniform(low=35, high=75, size=(NUM,))
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

    for i, frame in enumerate(tqdm.tqdm(sequence)):
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Cygnus_Real"], bpy.data.objects["Camera_Real"], bpy.data.objects["Sun"])
        frame.setup(bpy.data.scenes['Mask_ID'], bpy.data.objects["Cygnus_MaskID"], bpy.data.objects["Camera_MaskID"], bpy.data.objects["Sun"])

        # create name for the current image (unique to that image)
        name = shortuuid.uuid()
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
                                                                           bpy.data.objects['Cygnus_Real'], bpy.data.objects['Camera_Real'])
        og_keypoints = starfish.annotation.project_keypoints_onto_image(OG_KEYPOINTS.values(), bpy.data.scenes['Real'],
                                                                        bpy.data.objects['Cygnus_Real'], bpy.data.objects['Camera_Real'])
        frame.og_keypoints = {k: v for k, v in zip(OG_KEYPOINTS.keys(), og_keypoints)}

        frame.sequence_name = ds_name

        # dump data to json
        with open(os.path.join(output_node.base_path, "meta_0" + str(name)), "w") as f:
            f.write(frame.dumps())

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
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    print("...begining upload to %s..." % bucket_name)

    try:
        files = next(os.walk(os.path.join(os.getcwd(), "render", ds_name)))[2]
    except Exception:
        print("...No data set named " + ds_name + " found in starfish/render. Please generate images with that folder name or move existing folder into render folder")
        exit()
    # count number of files
    num_files = 0
    # For every file in directory
    for file in files:
        # ignore hidden files
        if not file.startswith('.'):
            # upload to s3
            print("uploading...")
            sys.stdout.write("\033[F")
            local_file = os.path.join(os.getcwd(), "render", ds_name, file)
            s3.upload_file(local_file, bucket_name, os.path.join(ds_name, file)
            num_files = num_files + 1

    print("...finished uploading...%d files uploaded..." % num_files)
    """


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
    generate(dataset_name)
    if runUpload in yes:
        upload(dataset_name, bucket_name)
    print("______________DONE EXECUTING______________")


if __name__ == "__main__":
    main()
