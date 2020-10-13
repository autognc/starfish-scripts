# starfish-scripts

See https://github.com/autognc/starfish.

This repo contains scripts used to generate labeled, synthetic imagery of Cygnus and Gateway by leveraging Starfish.

##  Scripts
1. __gen_cygnus_dataset.py:__
  For an example '.yaml' see __sample_config.yml__. This script is used to generate multiple imagesets one after another. 
  Each imageset can have an array of different augmentations. Great for creating datasets with multiple imagesets of various sizes with glare, blur, occlusion, or background randomization(or any combination of these augmentations). Images are labeled with bboxes and keypoints. NOTE: Background randomization technique depends on the .blend file used(see line 231 of script).
2. __Interpolated_cygnus_GB.py & Interpolated_dynamic.py:__ This script is used for creating interpolated image sequences with glare and blur of Cygnus and Gateway respectively.
3. __cygnus_RT.py:__ This script is used to render cygnus images with randomized textures.
4. __cygnus_keypointsGB.py:__ This script is used to render augmented cygnus images labeled with bboxes and keypoints. This script generates a single imageset, and has the same augmentation options as gen_cygnus_dataset.py
5. __cygnus_occlusion_old.py & cygnus_occlusion_new.py:__ these scripts were used for initial testing of generating occluded cygnus images. cygnus_occlusion_old.py generates labels with correct bboxes that go off the edge of the screen by cropping the final image after extracting the bbox from the mask. cygnus_occlusion_new.py uses the current technique for occlusion of achieving occlusion by setting offsets near the edge of the frame(included as an option in gen_cygnus_dataset.py).
6. __cygnus_keypoints.py:__ The base script for generating non-augmented cygnus images labeled with bboxes and keypoints. no augmentations are included in this script
7. __dynamic_moon.py:__ This script is used for generating images of gateway with dynamically sized moons, glare, blur, and domain-randomized-backgrounds
8. __SynImage_moon.py:__ This script was used to generate images of the moon from multiple distances and lighting angles used dynamicically-sized moon backgrounds
9. __cygnus_interpolated_keypoints.py:__ This script is used to generate non-augmented, interpolated image sequences of cygnus labeled with keypoints and bboxes
