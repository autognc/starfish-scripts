Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 
@mshah0686 
0
0 0 autognc/starfish
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Security  Insights  Settings
starfish/dsup.py
@mshah0686 mshah0686 Blend, dsup, synimage created
1cb803a 2 days ago
60 lines (47 sloc)  1.61 KB
    
"""
Author(s): Malav Shah
Upload Dataset to AWS
"""

import boto3
import os

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

def upload():
    print("\n\n______________STARTING UPLOAD_________")
    #input for data path
    localDataPath = raw_input("*> Enter Data Path: ") #r"/Users/Malav_Mac/Documents/Malav_Folder/TSL_Research/data_dump"
    #input for bucket name
    bucket_name = raw_input("*> Enter Bucket name: ") #'trialupload1'

    # Create an S3 client
    s3 = boto3.client('s3')

    #check if bucket name valid
    while not validate_bucket_name(bucket_name):
        bucket_name = raw_input("*> Enter Bucket name: ")

    #get all files within directory
    try:
        files =next(os.walk(localDataPath))[2]
    except:
        print("...Faulty data path. Run program again...")
        exit()
    print("...begining upload to %s..." % bucket_name) 
    
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

if __name__ == "__main__":
    upload()