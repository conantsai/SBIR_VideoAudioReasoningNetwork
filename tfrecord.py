import cv2 
from tensorflow.python.platform import gfile
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

def get_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def convert_videos_to_tfrecord(source_path, destination_path, n_videos_in_record, video_suffix, audio_suffix, width, height, tfrecord_name):

    # search image from video floder
    imagepaths = []
    audiopaths = []
    for path in os.listdir(source_path):
        imagename = gfile.Glob(os.path.join(source_path + path, video_suffix))
        imagepaths.append(imagename)
        audioname = gfile.Glob(os.path.join(source_path + path, audio_suffix))
        audiopaths.append(audioname)
    if not imagepaths:
        raise RuntimeError('No data files found.')
    print('Total videos found: ' + str(len(imagepaths)))

    imagepaths_split = list(get_chunks(imagepaths, n_videos_in_record))
    audiopaths_split = list(get_chunks(audiopaths, n_videos_in_record))

    
    if tfrecord_name == None:
        tfname = os.path.join(destination_path, "sample.tfrecords")
    else:
        tfname = os.path.join(destination_path, tfrecord_name + ".tfrecords")
    
    writer = tf.python_io.TFRecordWriter(tfname)

    for i, path in enumerate(imagepaths_split):
        for j, images in enumerate(path):
            image_feature = []

            # get audio for  each video name
            audio = audiopaths_split[i][j][0]
            npaudio = np.load(audio)

            # get label & score for  each video name
            slash_c = images[0].count("/")
            label = int(images[0].split("/", slash_c)[-2][6])
            if images[0].split("/", slash_c)[-2][8] == "0":
                score = int(images[0].split("/", slash_c)[-2][-2:])
            else:
                score = int(images[0].split("/", slash_c)[-2][-3:])
            
            # get all image for each video
            for image in images:
                image = cv2.imread(image)
                image = cv2.resize(image, (width, height))
                # plt.imshow(image) 
                # plt.show()
                image_feature.append(image)

            # convert to numpy array
            for k, feature in enumerate(image_feature):
                if k == 0:
                    image = np.array(feature)
                    npimage_feature = image
                else:
                    image = np.array(feature)
                    npimage_feature = np.concatenate((npimage_feature, image), axis=0)

            # encapsulate images, label, score
            example = tf.train.Example(features=tf.train.Features(feature={
                "images" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[npimage_feature.tostring()])),
                "audio" : tf.train.Feature(bytes_list=tf.train.BytesList(value=[npaudio.tostring()])),
                "label" : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "score" :  tf.train.Feature(int64_list=tf.train.Int64List(value=[score]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
    
    return

if __name__ == "__main__":
    convert_videos_to_tfrecord(source_path="/home/uscc/USAI_Outsourcing/B-1/Jinag_thesis/backup_thesis/fight_videos/testced/",
                               destination_path="/home/uscc/USAI_Outsourcing/B-1 tensorflow/train_record",
                               n_videos_in_record=16, video_suffix="*.jpg", audio_suffix="*.npy", width=242, height=182, tfrecord_name=None)