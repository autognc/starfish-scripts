import numpy as np
import bpy
import ssi
from ssi.rotations import Spherical
from mathutils import Euler
from ssi import utils
import json
import time
import os
import sys
import boto3
import uuid


def generate(ds_name):
    start_time = time.time()
    poses = utils.random_rotations(5)
    lightAngle = utils.random_rotations(3)
    #positions = utils.cartesian([0], [1, 2], [3,4,5])
    #offsets = utils.cartesian([0.25, 0.3, 0.46, 0.68, 0.8, 0.9], [0.3, 0.41, 0.56, 0.68, 0.75, 0.8, 0.93])
    
    seq = ssi.Sequence.exhaustive(
        #position = positions
        pose = poses,
        distance=[150],
        #lighting = lightAngle
        #offset = offsets
    )

    #check if dataset exists in render, if not, create folder
    try:
        os.mkdir("render/" + ds_name)
    except Exception:
        pass
    
    data_storage_path = os.getcwd() + "/render/" + ds_name 	
    
    #setting file output stuff
    output_node = bpy.data.scenes["Render"].node_tree.nodes["File Output"]
    output_node.base_path = data_storage_path
    
    #set black background
    bpy.context.scene.world.color = (0,0,0)
    
    #remove all animation
    for obj in bpy.context.scene.objects:
        obj.animation_data_clear()
        
    image_num = 0
        
    for i, frame in enumerate(seq):
        frame.setup(bpy.data.objects["Enhanced Cygnus"], bpy.data.objects["Camera1"], bpy.data.objects["Sun"])
        # add metadata to frame
        #frame.timestamp = int(time.time() * 1000)
        frame.sequence_name = ds_name
        
	
        bpy.context.scene.frame_set(0)
        # set output path
        name = uuid.uuid4()
        output_node.file_slots[0].path = str(name) + "#"
        output_node.file_slots[1].path = str(name) + "#_mask"
        
        # dump data to json
        with open(os.path.join(output_node.base_path, str(name) + "0_meta.json"), "w") as f:
            f.write(frame.dumps())
        image_num = i + 1
        # render
        #bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations = 1)
        bpy.ops.render.render(scene="Render")

    print("===========================================" + "\r")
    time_taken = time.time() - start_time
    print("------Time Taken: %s seconds----------" %(time_taken) + "\r")
    print("Number of images generated: " + str(image_num) + "\r")
    print("Total number of files: " + str(image_num * 3) + "\r")
    print("Average time per image: " + str(time_taken / image_num))
    print("Data stored at: " + data_storage_path)


def upload(ds_name, bucket_name):
    print("\n\n______________STARTING UPLOAD_________")

    # Create an S3 client
    s3 = boto3.client('s3')

    print("...begining upload to %s..." % bucket_name) 
    

    files =next(os.walk(os.cwd() + "/render/" + ds_name))[2]
    #cound number of files
    num_files = 0
    # For every file in directory
    for file in files:
        #ignore hidden files
        if not file.startswith('.'):
            #upload to s3
            local_file = os.path.join(localDataPath, file)
            s3.upload_file(local_file, bucket_name, file)
            num_files = num_files + 1

    print("...finished uploading...%d files uploaded..." % num_files)

def validate_bucket_name(bucket_name):
    s3t = boto3.resource('s3')
    #check if bucket exits. If not return false
    if s3t.Bucket(bucket_name).creation_date is None:
        print("...Bucket does not exits, enter valid bucket name...")
        return False
    else:
        #if exists, return true
        print("...bucket exists....")
        return True

if __name__ == "__main__":
    #check if render directory exists, if not create it
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
        	bucket_name = raw_input("*> Enter Bucket name: ")
    
    print("Note: if you want to upload to AWS but not generate images, move folder with images to 'render' and enter dataset name")
    print("Note: if the dataset name exists, images will be stored in that directory")
    dataset_name = input("*> Enter name for dataset: ")
    print("Note: rendered images will be stored in a directory called 'render' in the same local directory this script is located under the directory name you specify.")
    
    if runGen in yes:
    	generate(dataset_name)
    if runUpload in yes: 
    	upload(dataset_name, bucket_name)
