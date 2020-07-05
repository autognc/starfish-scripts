import numpy as np
import bpy
import starfish
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
def convert_to_float(i):
    return float(i)

LABEL_MAP = {
    'gateway': (206, 0, 206)
}

#********************************************************************************************
############################################
#The following is the main code for image generation
############################################
SCALE = 17
RES_X = 1024
RES_Y = 576
GLARE_TYPES = ['FOG_GLOW', 'SIMPLE_STAR', 'STREAKS', 'GHOSTS']
def generate(ds_name, tags_list, background_dir=None):
    start_time = time.time()

    #check if folder exists in render, if not, create folder
    try:
        os.mkdir("render/" + ds_name)
    except Exception:
        pass
    
    data_storage_path = os.getcwd() + "/render/" + ds_name     
    
    
    prop = bpy.context.preferences.addons['cycles'].preferences
    prop.get_devices()

    prop.compute_device_type = 'CUDA'

    for device in prop.devices:
        if device.type == 'CUDA':
            device.use = True
    
    bpy.context.scene.cycles.device = 'GPU'

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

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
        waypoints.append(starfish.Frame(
            distance=nm_to_bu(d['distance']),
            offset=d['offset'],
            background=Euler(list(map(deg_to_rad, d['background']))),
            pose=Euler(list(map(deg_to_rad, d['pose']))),
            lighting=Euler(list(map(deg_to_rad, d['lighting'])))
        ))

    counts = [300] * 6
    blur_vals = [(0,0)]
    glare_vals = [(0,5)]
    for i in range(1801//5):
        blur_x_y = np.random.uniform(2, 6, 2)
        glare_g_t = (np.random.randint(0,4), np.random.beta(2,8)*3)
        for j in range(0,5):
            blur_vals.append(blur_x_y)
            glare_vals.append(glare_g_t)
    for scene in bpy.data.scenes:
        scene.unit_settings.scale_length = 1 / SCALE
        
        
    if background_dir is not None:
        images_list = []
        img_names = []
        for f in os.listdir(background_dir):
            if f.endswith(".exr"):
                imgnum = f.split("_")[1]
                imgnum = imgnum.split(".")[0] + "." + imgnum.split(".")[1]
                images_list.append(f)
                img_names.append(imgnum)
        images_list = sorted(images_list)
        img_names = sorted(img_names, key=convert_to_float)
        moon_num = 0
        
    else:
        img_names = []
    moon_count = 0;
    # first few moons are too big
    img_names = img_names[12:]
    print(img_names)
    num_moons = len(img_names)
    print(num_moons)
    for i, frame in enumerate(starfish.Sequence.interpolated(waypoints, counts)):
        bpy.context.scene.frame_set(0)
        frame.setup(bpy.data.scenes['Real'], bpy.data.objects["Gateway"], bpy.data.objects["Camera"], bpy.data.objects["Sun"])
    
        #create name for the current image (unique to that image)
        name = str(i).zfill(5)
        output_node.file_slots[0].path = "image_" + "#" + str(name)
        output_node.file_slots[1].path = "mask_" + "#" + str(name)

        bpy.data.scenes["Render"].node_tree.nodes["Blur"].size_x = blur_vals[i][0] 
        bpy.data.scenes["Render"].node_tree.nodes["Blur"].size_y = blur_vals[i][1] 
        
        glare_value = 0.5
        bpy.data.scenes["Render"].node_tree.nodes["Glare"].glare_type = GLARE_TYPES[glare_vals[i][0]]
        bpy.data.scenes["Render"].node_tree.nodes["Glare"].mix = glare_value
        bpy.data.scenes["Render"].node_tree.nodes["Glare"].threshold = glare_vals[i][1]       
        
        
        # load new Environment Texture
        if img_names:
            moon_name = img_names[moon_num]
            image = bpy.data.images.load(filepath = background_dir + "/image_" + moon_name + ".exr")
            bpy.data.worlds["World"].node_tree.nodes['Environment Texture'].image = image
            moon_count += 1
            if moon_count > (1800//num_moons+1):
                moon_num = (moon_num + 1) % num_moons
                moon_count = 0;    
        # render
        bpy.ops.render.render(scene="Render")
        
        # Tag the pictures
        frame.tags = tags_list
        # add metadata to frame
        frame.sequence_name = ds_name

        mask_filepath = os.path.join(output_node.base_path, "mask_0" + str(name) + ".png")
        meta_filepath = os.path.join(output_node.base_path, "meta_0" + str(name) + ".json")

        # run color normalization with labels plus black background
        normalize_mask_colors(mask_filepath, list(LABEL_MAP.values()) + [(0, 0, 0)])

        # get bbox and centroid and add them to metadata
        frame.bboxes = get_bounding_boxes_from_mask(mask_filepath, LABEL_MAP)
        frame.centroids =  get_centroids_from_mask(mask_filepath, LABEL_MAP)
    
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
    
