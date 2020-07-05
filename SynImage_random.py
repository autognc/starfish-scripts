import numpy as np
import bpy
import starfish
from starfish.rotations import Spherical
from starfish.annotation import mask
from mathutils import Euler
import starfish.annotation
from starfish.annotation import get_bounding_boxes_from_mask, get_centroids_from_mask, normalize_mask_colors
from starfish import utils
import json
import math
import time
import os
import sys
import boto3
import shortuuid
import csv
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
NUM = 1000
SCALE = 17
MOON_RADIUS = 0.4
MOON_CENTERX = 4.723
MOON_CENTERY = 0
RES_X = 1024
RES_Y = 576

def generate(ds_name, tags_list, background_dir=None):
    start_time = time.time()

    

    prop = bpy.context.preferences.addons['cycles'].preferences
    prop.get_devices()

    prop.compute_device_type = 'CUDA'

    for device in prop.devices:
        if device.type == 'CUDA':
            device.use = True
    
    bpy.context.scene.cycles.device = 'GPU'

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'


    #check if folder exists in render, if not, create folder
    try:
        os.mkdir("render/" + ds_name)
    except Exception:
        pass
    
    data_storage_path = os.getcwd() + "/render/" + ds_name     
    #setting file output stuff
    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path
    
    #set black background
    #bpy.context.scene.world.color = (0,0,0)
    
    #remove all animation
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            obj.animation_data_clear()
        
    image_num = 0
    shortuuid.set_alphabet('12345678abcdefghijklmnopqrstwxyz')
    # np.random.seed(42)
    poses = utils.random_rotations(NUM)
    lightings = utils.random_rotations(NUM)
    moon_num = 0
    # check if background dir is not None and get list of .exr files in that directory
    if background_dir is not None:
        images_list = []
        for f in os.listdir(background_dir):
            if f.endswith(".exr") or f.endswith(".jpg") or f.endswith(".png"):
                images_list.append(f)
        images_list = sorted(images_list)
        num_images = len(images_list)
        moon_num = 0
        
    else:
        num_images = 0

    #THIS SCRIPT GENERATES IMAGES WITH ZERO GLARE AND ZERO BLUR
    glare_value = -1
    glare_threshold = 8
    bpy.data.scenes["Render"].node_tree.nodes["Glare"].mix = glare_value
    bpy.data.scenes["Render"].node_tree.nodes["Glare"].threshold = glare_threshold
        
    #set blur values
    blur_x = 0
    blur_y = 0
    bpy.data.scenes["Render"].node_tree.nodes["Blur"].size_x = blur_x
    bpy.data.scenes["Render"].node_tree.nodes["Blur"].size_y = blur_y

    for i, (pose, lighting) in enumerate(zip(poses, lightings)):
    
        for scene in bpy.data.scenes:
            scene.unit_settings.scale_length = 1 / SCALE

        nmi = np.random.uniform(low=0.5, high=6)
        distance = nmi * 1852 * SCALE
        
        # rotates camera to prevent showing blender environment texture wrapping weirdness
        r = MOON_RADIUS/(moon_num+1) * np.sqrt(np.random.random())
        t = np.random.uniform(low=0, high=2 * np.pi)


        background = Euler([0, MOON_CENTERX - r * np.cos(t), MOON_CENTERY + r * np.sin(t)])


        offset = np.random.uniform(low=0.0, high=1.0, size=(2,))
        
        bpy.context.scene.frame_set(0)
        frame = starfish.Frame(
            background=background,
            pose=pose,
            lighting=lighting,
            distance=distance,
            offset=offset
        )
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Gateway"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])

        # load new Environment Texture
        if num_images != 0:
            image = bpy.data.images.load(filepath = os.getcwd()+ '/' + background_dir + '/' + images_list[moon_num])
            bpy.data.worlds["World"].node_tree.nodes['Environment Texture'].image = image
            moon_num = moon_num + 1 if moon_num < num_images-1 else 0

        #create name for the current image (unique to that image)
        name = shortuuid.uuid() 
        output_node.file_slots[0].path = "image_"+ str(name) + "#"
        output_node.file_slots[1].path = "mask_" + str(name) + "#"

        mask_filepath = os.path.join(output_node.base_path, "mask_" + str(name) + "0.png")
        meta_filepath = os.path.join(output_node.base_path, "meta_" + str(name) + "0.json")
        
        # render
        bpy.ops.render.render(scene="Render")
        
        #Tag the pictures
        frame.tags = tags_list
        # add metadata to frame
        frame.sequence_name = ds_name
        frame.glare_value = glare_value

        # run color normalization with labels plus black background
        normalize_mask_colors(mask_filepath, list(LABEL_MAP.values()) + [(0, 0, 0)])

        # get bbox and centroid and add them to metadata
        frame.bboxes = get_bounding_boxes_from_mask(mask_filepath, LABEL_MAP)
        frame.centroids =  get_centroids_from_mask(mask_filepath, LABEL_MAP)
        
        with open(os.path.join(output_node.base_path, "meta_" + str(name) + "0.json"), "w") as f:
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

    # prompt user for directory of background images
    background_sequence = input("*> Would you like to use mutliple background images?[y/n]: ")
    if background_sequence in yes:
        background_dir = input("*> Enter Image Directory: ")
        while not os.path.isdir(background_dir):
            background_dir = input("*> Enter Image Directory: ")
    
    tags_list = tags.split();
    if runGen in yes:
        if background_sequence in yes:
            generate(dataset_name, tags_list, background_dir)
        else:
            generate(dataset_name, tags_list)
    if runUpload in yes: 
        upload(dataset_name, bucket_name)
    print("______________DONE EXECUTING______________")

if __name__ == "__main__":
    main()
    
