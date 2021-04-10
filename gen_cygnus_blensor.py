import glob
import json
import os
import shutil
import subprocess
import time

import blensor
import boto3
import bpy
import numpy as np
import starfish
import starfish.annotation
import tqdm
from mathutils import Euler, Quaternion

"""
    0: timestamp 
    1: yaw, 
    2:pitch
    3:distance,
    4:distance_noise
    5-7: x,y,z
    8-10:x_noise,y_noise,z_noise
    11:object_id
    12:255*color[0]
    13:255*color[1]
    14:255*color[2]
    15:idx
"""


def generate(ds_name, bucket, tags):

    """for i, frame in enumerate(sequence):
    frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Cygnus_Real"], bpy.data.objects["Camera_Real"], bpy.data.objects["Sun"])

    scanner = bpy.data.objects["Camera_Real"]
    blensor.blendodyne.scan_advanced(scanner, rotation_speed = 5.0,
                            simulation_fps=24, angle_resolution = 0.12,
                            max_distance = 120, evd_file= f"/home/jdeutsch/blensor_1.0.18-RC10_x64/{os.basename(fname)}.numpy",
                            noise_mu=0.0, noise_sigma=0.03, start_angle = 0.0,
                            end_angle = 360.0,
                            add_blender_mesh = False,
                            add_noisy_blender_mesh = False)"""
    """blensor.blendodyne.scan_advanced(scanner, rotation_speed = 5.0, 
                    simulation_fps=24, angle_resolution = 0.12, 
                    max_distance = 120, evd_file= f"/home/jdeutsch/blensor_1.0.18-RC10_x64/{os.path.basename(meta).split('.')[0]}.numpy",
                    noise_mu=0.0, noise_sigma=0.03, start_angle = 0.0, 
                    end_angle = 360.0, 
                    add_blender_mesh = False, 
                    add_noisy_blender_mesh = False)"""
    start_time = time.time()

    # check if folder exists in render, if not, create folder
    try:
        os.mkdir(os.path.join("render", ds_name))
    except Exception:
        pass

    data_storage_path = os.path.join(os.getcwd(), "render", ds_name)

    # copy script and write to data storage path
    with open(os.path.abspath(__file__), 'r') as f:
        code = f.read()
    with open(os.path.join(data_storage_path, 'gen_code_blensor.py'), 'w') as f:
        f.write(code)
   
    download_meta(ds_name, bucket)
    metas = sorted(glob.glob(os.path.join(data_storage_path, "meta_*")))
    scanner_type = "tof"
    for meta in tqdm.tqdm(metas):
        with open(meta, "r") as f:
            info = json.load(f)
        print(meta)
        uuid = os.path.basename(meta).split(".")[0][5:]
        print(uuid)
        frame = starfish.Frame(
            pose=Quaternion(info["pose"]),
            lighting=Quaternion(info["lighting"]),
            background=Quaternion([1,0,0,0]),
            distance=info["distance"],
            offset=info["offset"],
        )
        frame.setup(
            bpy.data.scenes["Real"],
            bpy.data.objects["Cygnus_Real"],
            bpy.data.objects["Camera_Real"],
            bpy.data.objects["Sun"],
        )

        scanner = bpy.data.objects["Camera_Real"]

        output_path = os.path.join(data_storage_path, f"lidar_{uuid}.numpy")

        blensor.tof.scan_advanced(
            scanner,
            max_distance=200,
            evd_file=output_path,
            noise_mu=0.0,
            noise_sigma=0.1,
            tof_res_x=176,
            tof_res_y=144,
            lens_angle_w=43.6,
            lens_angle_h=34.6,
            flength=10.0,
            add_blender_mesh=False,
            add_noisy_blender_mesh=False,
        )
        blensor_renamed = os.path.join(data_storage_path, f"lidar_{uuid}00000.numpy")
        shutil.move(blensor_renamed, output_path)
        info["lidar_tags"] = tags
        with open(meta, "w") as f:
            f.write(json.dumps(info, indent=2))
            f.write('\n')
    upload(ds_name, bucket)
    print(f"Done! Took: {time.time()-start_time} seconds.")


def download_meta(ds_name, bucket_name):
    print("\n\n______________STARTING DOWNLOAD_________")

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"s3://{bucket_name}/{ds_name}",
            os.path.join("render", ds_name),
            "--exclude", "*",
            "--include", "meta_*",
        ],
        check=True # with out check=True, downloads all files
    )


def upload(ds_name, bucket_name):
    print("\n\n______________STARTING UPLOAD_________")

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            os.path.join("render", ds_name),
            f"s3://{bucket_name}/{ds_name}",
        ]
    )


def validate_bucket_name(bucket_name):
    s3t = boto3.resource("s3")
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

    yes = {"y", "Y", "yes"}
    bucket_name = input("*> Enter Bucket name: ")
    # check if bucket name valid
    while not validate_bucket_name(bucket_name):
        bucket_name = input("*> Enter Bucket name: ")

    dataset_name = input("*> Enter name of Imageset: ")
    print(
        "   Note: rendered images will be stored in a directory called 'render' in the same local directory this script is located under the directory name you specify."
    )
    tags = input("*> Enter tags for the batch seperated with space: ")
    tags_list = tags.split()

    generate(dataset_name, bucket_name, tags_list)
    print("______________DONE EXECUTING______________")


if __name__ == "__main__":
    main()
