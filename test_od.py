import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import argparse

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from moviepy.editor import VideoFileClip

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

NUM_CLASSES = 90

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def process_image(image_np):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_object(image_np, sess, detection_graph, category_index)
            return image_process      

def detect_object(image_np, sess, detection_graph, category_index):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)    
    return image_np

def main(flags):
    # What model to download.
    MODEL_NAME = flags.model_name
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    
    if os.path.exists(PATH_TO_CKPT):
        print ("%s exist" % PATH_TO_CKPT)
    else:
        print ("%s download" % PATH_TO_CKPT)
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())


    global detection_graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = flags.model_dir + '/object_detection/data/mscoco_label_map.pbtxt'
    if os.path.exists(PATH_TO_LABELS):
        print ("%s exist" % PATH_TO_LABELS)
    else:
        print ("%s download" % PATH_TO_LABELS)
        return

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    global category_index  
    category_index = label_map_util.create_category_index(categories)        

    if flags.cmd == 'image':
        PATH_TO_TEST_IMAGES_DIR = flags.image_dir
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 6) ]
        print ("image path: %s" % TEST_IMAGE_PATHS)

        # Size, in inches, of the output images.
        IMAGE_SIZE = (12, 8)
        
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)         
            image_np = load_image_into_numpy_array(image)
            image_result = process_image(image_np)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_result)        
        plt.show()       

    elif flags.cmd == 'video':
        PATH_TO_TEST_VIDEOS_DIR = flags.video_dir
        TEST_VIDEO_PATHS = [ os.path.join(PATH_TO_TEST_VIDEOS_DIR, 'skate.mp4') ]
        print ("video path: %s" % TEST_VIDEO_PATHS)

        white_output = PATH_TO_TEST_VIDEOS_DIR + '/skate_out.mp4'
        clip = VideoFileClip(os.path.join(PATH_TO_TEST_VIDEOS_DIR, 'skate.mp4')).subclip(0,10)
        white_clip = clip.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cmd',
        type=str,
        default='image',
        help='work style to choose, image or video.'
    )            
    parser.add_argument(
        '--model_name',
        type=str,
        default='ssd_mobilenet_v1_coco_11_06_2017',
        help='model to choose.'
    )        
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/Users/baixiao/Go/src/github.com/tensorflow/models',
        help='Directory to load object-detection model.'
    )    
    parser.add_argument(
        '--image_dir',
        type=str,
        default='test_images',
        help='Directory to locate images.'
    )        
    parser.add_argument(
        '--video_dir',
        type=str,
        default='test_videos',
        help='Directory to locate images.'
    )        
    parser.add_argument(
        '--export_dir',
        type=str,
        default='./export',
        help='Directory to export model.'
    )
    parser.add_argument(
        '--model_version',
        type=int,
        default=1,
        help='Model version'
    )

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)