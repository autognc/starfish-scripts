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

def main():
    start_time = time.time()
    data_storage_path = "/home/mhs2466/Documents/TSL/render"
    poses = utils.random_rotations(5)
    lightAngle = utils.random_rotations(3)
    #positions = utils.cartesian([0], [1, 2], [3,4,5])
    #offsets = utils.cartesian([0.25, 0.3, 0.46, 0.68, 0.8, 0.9], [0.3, 0.41, 0.56, 0.68, 0.75, 0.8, 0.93])
    
    seq = ssi.Sequence.exhaustive(
        #position = positions
        pose = poses,
        distance=[100, 150, 200],
        lighting = lightAngle
        #offset = offsets
    )
    
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
        frame.timestamp = int(time.time() * 1000)
        frame.sequence_name = "Render Testing"
        
        # set output path
        output_node.file_slots[0].path = f"real#_{i}"
        output_node.file_slots[1].path = f"mask#_{i}"
        
        # dump data to json
        with open(os.path.join(output_node.base_path, f"{i}.json"), "w") as f:
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


if __name__ == "__main__":
    yes = {'yes', 'y', 'Y'}
    runGen = input("Generate images?[y/n]: ")
    if runGen in yes:
    	main()
    
    runUpload = input("Would you like to upload these images to AWS? [y/n]: ")
    if runUpload in yes: 
    	os.system('python dsup.py')
    
#### MODIFYING SUN INTENSITY
###IF USING CYCLES RENDER: MUST MODIFY THROUGH USING NODES and CHANGE STRENGTH
###IF USING EVEEE RENDER: bpy.data.lights['Sun'].energy = _______
