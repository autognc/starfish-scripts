import numpy as np
import bpy
import ssi
from mathutils import Euler
from ssi.core import utils
from ssi import postprocessing
import json
import math
import time
import os
import sys
import boto3
import shortuuid
import csv
from PIL import Image
from collections import defaultdict
import random

def nm_to_bu(nmi):
    return nmi * 1852 * SCALE  # convert from nmi to blender units

def deg_to_rad(deg):
    return deg * np.pi / 180  # convert from degrees to radians

LABEL_MAP = {
    'gateway': (206, 0, 206)
}

#********************************************************************************************
############################################
#The following is the main code for image generation
############################################
SCALE = 17
def generate(ds_name, tags_list):
    start_time = time.time()

    #check if folder exists in render, if not, create folder
    try:
        os.mkdir("render/" + ds_name)
    except Exception:
        pass
    
    data_storage_path = os.getcwd() + "/render/" + ds_name     
    
    # switch to correct scene
    bpy.context.window.scene = bpy.data.scenes['Real']

    # remove all animation
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()

    # set up file outputs
    output_node = bpy.data.scenes['Render'].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path
        
    np.random.seed(5)
    waypoints_dict = {
        'distance': [#0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
            6,
            3,
            1,
            
            3,
            6,
            
            3,
            1
        ]
    ,
        'offset': [
            (0.25, 0.25),
            (0.5, 0.5),
            (0.75, 0.75),
        
            (0.7, 0.3),
            (0.5, 0.25),
        
            (0.35, 0.5),
            (0.25, 0.5)
        ],
        'background': [
            (0, -90, 0),
            (0, -105, 15),
            (0, -120, 30),
            
            (0, -120, 15),
            (0, -120, 0),
        
            (0, -105, 0),
            (0, -90, 0)
        ],
        'pose': np.random.rand(7, 3) * 360,
        'lighting': np.random.rand(7, 3) * 360
    }

    waypoints = []
    for vals in zip(*waypoints_dict.values()):
        d = dict(zip(waypoints_dict.keys(), vals))
        waypoints.append(ssi.Frame(
            distance=nm_to_bu(d['distance']),
            offset=d['offset'],
            background=Euler(list(map(deg_to_rad, d['background']))),
            pose=Euler(list(map(deg_to_rad, d['pose']))),
            lighting=Euler(list(map(deg_to_rad, d['lighting'])))
        ))

    counts = [300] * 7

    for scene in bpy.data.scenes:
        scene.unit_settings.scale_length = 1 / SCALE

    for i, frame in enumerate(ssi.Sequence.interpolated(waypoints, counts)):
        bpy.context.scene.frame_set(0)
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Gateway"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])
    
        #create name for the current image (unique to that image)
        name = str(i).zfill(5)
        output_node.file_slots[0].path = "image_" + "#" + str(name)
        output_node.file_slots[1].path = "mask_" + "#" + str(name)
        
        # render
        bpy.ops.render.render(scene="Render")
        
        # Tag the pictures
        frame.tags = tags_list
        # add metadata to frame
        frame.sequence_name = ds_name

        mask_filepath = os.path.join(output_node.base_path, "mask_0" + str(name) + ".png")
        meta_filepath = os.path.join(output_node.base_path, "meta_0" + str(name) + ".json")

        # run color normalization with labels plus black background
        postprocessing.normalize_mask_colors(mask_filepath, list(LABEL_MAP.values()) + [(0, 0, 0)])

        # get bbox and centroid and add them to metadata
        frame.bboxes = postprocessing.get_bounding_boxes_from_mask(mask_filepath, LABEL_MAP)
        frame.centroids = postprocessing.get_centroids_from_mask(mask_filepath, LABEL_MAP)
    
        with open(meta_filepath, "w") as f:
            f.write(frame.dumps())

    print("===========================================" + "\r")
    time_taken = time.time() - start_time
    print("------Time Taken: %s seconds----------" %(time_taken) + "\r")
    print("Data stored at: " + data_storage_path)
    bpy.ops.wm.quit_blender()

############################
#The following is the main code for upload
############################
def upload(ds_name, bucket_name):
    print("\n\n______________STARTING UPLOAD_________")

    # Create an S3 client
    s3 = boto3.client('s3')

    print("...begining upload to %s..." % bucket_name) 
    
    try:
        files =next(os.walk(os.getcwd() + "/render/" + ds_name))[2]
    except Exception:
        print("...No data set named " + ds_name + " found in starfish/render. Please generate images with that folder name or move existing folder into render folder")
        exit()
    #count number of files
    num_files = 0
    # For every file in directory
    for file in files:
        #ignore hidden files
        if not file.startswith('.') and not file.startswith('truth'):
            #upload to s3
            print("uploading...")
            sys.stdout.write("\033[F")
            local_file = os.path.join(os.getcwd() + "/render/" + ds_name, file)
            s3.upload_file(local_file, bucket_name, ds_name + "/" + file)
            num_files = num_files + 1

    print("...finished uploading...%d files uploaded..." % num_files)

def validate_bucket_name(bucket_name):
    s3t = boto3.resource('s3')
    #check if bucket exits. If not return false
    if s3t.Bucket(bucket_name).creation_date is None:
        print("...Bucket does not exist, enter valid bucket name...")
        return False
    else:
        #if exists, return true
        print("...bucket exists....")
        return True
#############################################
#Run user input data then run generation/upload
#############################################
def main():
    try:
        os.mkdir("render")
    except Exception:
        pass

    yes = {'yes', 'y', 'Y'}
    runGen = input("*> Generate images?[y/n]: ")
    
    runUpload = input("*> Would you like to upload these images to AWS? [y/n]: ")
    if runUpload in yes: 
        bucket_name = input("*> Enter Bucket name: ")
        #check if bucket name valid
        while not validate_bucket_name(bucket_name):
            bucket_name = input("*> Enter Bucket name: ")
    
    print("   Note: if you want to upload to AWS but not generate images, move folder with images to 'render' and enter folder name. If the folder name exists, images will be stored in that directory")
    dataset_name = input("*> Enter name for folder: ")
    print("   Note: rendered images will be stored in a directory called 'render' in the same local directory this script is located under the directory name you specify.")
    tags = input("*> Enter tags for the batch seperated with space: ")
    tags_list = tags.split();
    if runGen in yes:
        generate(dataset_name, tags_list)
    if runUpload in yes: 
        upload(dataset_name, bucket_name)
    print("______________DONE EXECUTING______________")

if __name__ == "__main__":
    main()
    
