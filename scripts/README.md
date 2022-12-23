# Runing single image inference

To run inference on a single image make sure you change the input sample's dimensions along with the intrinsic parameters in detect.py. 

```python
def input_sample(image_path):   #partial copy of datagen._parse_and_preprocess()

    # input image converted into tensor
    input_image = Image.open(image_path)
    im = tf.convert_to_tensor(input_image,dtype=tf.float32) 

    # input image dimensions
    im_h_orig = tf.cast(720, tf.int32)
    im_w_orig = tf.cast(1280, tf.int32)

    im_h_new = tf.minimum(FLAGS.infer_max_height_before_crop, im_h_orig)
    im_scale = tf.cast(im_h_new, tf.float32) / tf.cast(im_h_orig, tf.float32)
    im_w_new = tf.cast(tf.cast(im_w_orig, tf.float32) * im_scale, tf.int32)

    # croping size for inference
    crop_w = FLAGS.infer_crop_size[0]
    crop_h = FLAGS.infer_crop_size[1]

    im = misc.resize_image_tf(im, (1280, 720))
    im.set_shape([720, 1280, 3])

    # K matrix 
    # !!!FOR THE CROPED IMAGE!!!
    fx = 640.9191284179688 #* im_scale
    fy = 639.4614868164062 #* im_scale
    cx = 631.1490478515625 #* im_scale
    cy = 363.1187744140625 #* im_scale

    K = tf.convert_to_tensor(
      [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], name='K')

    # define the output dictionary sample  
    # scene id set to 1 since we don't care about it we only infer one image
    # same thing for the IM_ID
    sample = {
      common.SCENE_ID: tf.convert_to_tensor(1,dtype=tf.int64),
      common.IM_ID: tf.convert_to_tensor(1,dtype=tf.int64),
      common.IMAGE_PATH: tf.convert_to_tensor(image_path,dtype=tf.string),
      common.IMAGE: im,
      common.K: K,
    }
    # note that the sample dictionary cotains more tensors regarding the gt poses and annotations
    # since we care about the inference they have been removed
    return sample
```
    

Then you just run it with:

```python detect.py --model= ModelName --vis (optional)```
