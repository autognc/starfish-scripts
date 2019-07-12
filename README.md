# starfish
Use `git status` to list all new or modified files that haven't yet been committed.


Setup
- 
 - modules needed: boto3, bpy, ssi, numpy, mathutils

Running Script
- 
 - in the directory with the blender file, run: 
 'blender [Blend File Name] -b -P SynImage.py'
 - This will ask you a series of prompts. The images will be stored inside the 'render' folder that will be created in the starfish directory. 
 - For uploading, make sure the aws credentials are set up. The script will access your buckets and create a similar directory as that of the folder in render.
 
Types of Images that can be created (variables that can be modified for a picture):
- 
 - **Camera Distance:** The distance between the cygnus and camera. Will make cygnus closer or farther in the scene.
 
 - **Light Angle:** Specify the different angles within a 3D space that create different lighting directions on cygnus. This does not change the light intensity (there is no variable for light intensity yet, but we can manually change it in blender). Rather, think of it as positioning the sun at a different location on a sphere with cygnus at the center of the sphere (e.g. from above, from the right, from behind, etc.). Right now we randomly create different positions.
 
 - **Pose:** Pose is the rotation of cygnus, so the 6 degree of freedom. Right now we randomly create different pose/rotations.
 
 - **Offset:** Offset defines the position of cygnus within the scene (i.e. if the scene were a x and y plane, offset defines the different coordinates cygnus will be at). Allows you to basically move cygnus to different areas within the scene rather than the center.
 
 - **Object Position:** The absolute 3D position of the object in the global coordinate system. This is for when we have a background (so if earth is in the background or not) so not being used right now. 
 
 - **Background/camera orientation:** the orientation of the camera relative to the global coordinate system. This affects only what part of the scene appears in the background and at what angle it appears.
