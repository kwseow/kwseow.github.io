---
layout: default
---

Happy National day, Singapore!
# Public Datasets
+ [Dataset Finder](#dataset-finder)

<center><img src="images/DLI Header.png" alt="Header" width="400"></center>

# 1.0 Object Detection Application
The DeepStream SDK offers a complete set of sample reference applications and pre-trained neural networks to jump-start development.  In this lab, you'll work with the `deepstream-test1` reference application to find objects in a video stream, annotate them with bounding boxes, and output the annotated stream along with a count of the objects found.

<img src="images/01_threethingsio.png">

You'll follow the steps below to build your own applications based on the reference app:

1.1 **[Build a Basic DeepStream Pipeline](#01_overview)**<br>
&nbsp; &nbsp; &nbsp; 1.1.1 [Sample Application `deepstream-test1`](#test1)<br>
&nbsp; &nbsp; &nbsp; 1.1.2 [Sample Application plus RTSP - `deepstream-test1-rtsp-out`](#rtsp)<br>
&nbsp; &nbsp; &nbsp; 1.1.3 [Exercise: Build and Run the Base Application](#01_ex_base)<br>
1.2 **[Configure an Object Detection Model](#01_change_objects)**<br>
&nbsp; &nbsp; &nbsp; 1.2.1 [Gst-nvinfer Configuration File](#01_config)<br>
&nbsp; &nbsp; &nbsp; 1.2.2 [Exercise: Detect Only Two Object Types](#01_ex_change)<br>
1.3 **[Modify Metadata to Perform Analysis](#01_count_objects)**<br>
&nbsp; &nbsp; &nbsp; 1.3.1 [Extracting Metadata with a GStreamer Probe](#01_probe)<br>
&nbsp; &nbsp; &nbsp; 1.3.2 [Exercise: Count Vehicles and Bikes](#01_ex_count)<br>
1.4 **[Put It All Together](#01_final)**<br>
&nbsp; &nbsp; &nbsp; 1.4.1 [Exercise: Detect and Count three Object Types](#01_ex_challenge)<br>

<a name='01_overview'></a>
# 1.1 Build a Basic DeepStream Pipeline
The framework used to build a DeepStream application is a GStreamer **pipeline** consisting of a video input stream, a series of **elements** or **plugins** to process the stream, and an insightful output stream. Each plugin has a defined input, also called its **sink**, and defined output, known as its **source**.  In the pipeline, the source pad of one plugin connects to the sink pad of the next in line.  The source includes information extracted from the processing, the **metadata**, which can be used for annotation of the video and other insights about the input stream.      

<img src="images/01_building_blocks.png">

<a name='test1'></a>
## 1.1.1 Sample Application - `deepstream-test1`
The DeepStream SDK includes plugins for building a pipeline, and some reference test applications. For example, the `deepstream_test1` application can take a street scene video file as input, use object detection to find vehicles, people, bicycles, and road signs within the video, and output a video stream with bounding boxes around the objects found.

<img src="images/01_exampleio2.png">

The reference test applications are in the `deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/` directory.  You can take a look at the C code for the `deepstream-test1` app at [deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test1/deepstream_test1_app.c](../deepstream_sdk_v4.0.2_jetson/sources/apps/sample_apps/deepstream-test1/deepstream_test1_app.c)<br><br>
Looking at the code, we can find where all the plugins are instantiated in `main` using the `gst_element_factory_make` method.  This is a good way to see exactly which plugins are in the pipeline *(Note: the sample snippets below are abbreviated code for clarity purposes)*:

```c
...
  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("dstest1-pipeline");

  /* Source element for reading from the file */
  source = gst_element_factory_make ("filesrc", "file-source");

  /* Since the data format in the input file is elementary h264 stream,
   * we need a h264parser */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");

  /* Use nvdec_h264 for hardware accelerated decode on GPU */
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  /* Use nvinfer to run inferencing on decoder's output,
   * behaviour of inferencing is set through config file */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* Use convertor to convert from NV12 to RGBA as required by nvosd */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");

  /* Create OSD to draw on the converted RGBA buffer */
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* Finally render the osd output */
  transform = gst_element_factory_make ("nvegltransform", "nvegl-transform");
  sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
...
```

We see that the input is a file source, `filesrc`, in H.264 video format, which is decoded and then run through the `nvinfer` inference engine to detect objects.  A buffer is created with `nvvideoconvert` so that bounding boxes can be overlaid on the video images with the `nvdsosd` plugin.  Finally, the output is rendered.

<a name='rtsp'></a>
## 1.1.2 Sample Application plus RTSP - `deepstream-test1-rtsp-out`
For the purposes of this lab, which runs headless on a Jetson Nano connected to a laptop, the video stream must be converted to a format that can be transferred to the laptop media player.  This is accomplished by customizing the sample app with additional plugins and some logic. Some specific customized apps are included in this lab in the `dli_apps` directory.  Take a look at the C code in [/home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out/deepstream_test1_app.c](../deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out/deepstream_test1_app.c).<br><br>
Scrolling down to `main`, we can see that there are a few differences in the rendering plugins used for the RTSP protocol transfer of the video stream *(Note: the sample snippets below are abbreviated code for clarity purposes)*:

```c
...
  /* Finally render the osd output */
  transform = gst_element_factory_make ("nvvideoconvert", "transform");
  cap_filter = gst_element_factory_make ("capsfilter", "filter");
  caps = gst_caps_from_string ("video/x-raw(memory:NVMM), format=I420");
  g_object_set (G_OBJECT (cap_filter), "caps", caps, NULL);

  encoder = gst_element_factory_make ("nvv4l2h264enc", "h264-encoder");
  rtppay = gst_element_factory_make ("rtph264pay", "rtppay-h264");

  g_object_set (G_OBJECT (encoder), "bitrate", 4000000, NULL);

#ifdef PLATFORM_TEGRA
  g_object_set (G_OBJECT (encoder), "preset-level", 1, NULL);
  g_object_set (G_OBJECT (encoder), "insert-sps-pps", 1, NULL);
  g_object_set (G_OBJECT (encoder), "bufapi-version", 1, NULL);
#endif
  sink = gst_element_factory_make ("udpsink", "udpsink");
...
```

The plugins are put in a pipeline bin with the `gst_bin_add_many()` methods  :

```c
...
  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, transform, cap_filter, encoder, rtppay, sink, NULL);
...
```

Next, a sink pad (input) for the `streammux` element is created and linked to the `decoder` source pad (output):

```c
...
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[16] = "sink_0";
  gchar pad_name_src[16] = "src";

  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  if (!sinkpad) {
    g_printerr ("Streammux request sink pad failed. Exiting.\n");
    return -1;
  }

  srcpad = gst_element_get_static_pad (decoder, pad_name_src);
  if (!srcpad) {
    g_printerr ("Decoder request src pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to stream muxer. Exiting.\n");
      return -1;
  }

...
```

Finally, the elements are linked together using the `gst_element_link_many()` method.  The start of the pipeline through the `decoder` are linked together, and the `streammux` and beyond are linked together, to form the entire pipeline.

```c
...
  /* we link the elements together */
  /* file-source -> h264-parser -> nvh264-decoder ->
   * nvinfer -> nvvidconv -> nvosd -> video-renderer */

  if (!gst_element_link_many (source, h264parser, decoder, NULL)) {
    g_printerr ("Elements could not be linked: 1. Exiting.\n");
    return -1;
  }

  if (!gst_element_link_many (streammux, pgie,
      nvvidconv, nvosd, transform, cap_filter, encoder, rtppay, sink, NULL)) {
    g_printerr ("Elements could not be linked: 2. Exiting.\n");
    return -1;
  }

...
```

In summary, the pipeline for this app consists of the following plugins (ordered):

- `GstFileSrc` - reads the video data from file
- `GstH264Parse` - parses the incoming H264 stream
- `Gst-nvv4l2decoder` - hardware accelerated decoder; decodes video streams using NVDEC
- `Gst-nvstreammux` - batch video streams before sending for AI inference
- `Gst-nvinfer` - runs inference using TensorRT
- `Gst-nvvideoconvert` - performs video color format conversion (I420 to RGBA)
- `Gst-nvdsosd` - draw bounding boxes, text and region of interest (ROI) polygons
- `Gst-nvvideoconvert` - performs video color format conversion (RGBA to I420)
- `GstCapsFilter` - enforces limitations on data (no data modification)
- `Gst-nvv4l2h264enc` - encodes RAW data in I420 format to H264
- `GstRtpH264Pay` - converts H264 encoded Payload to RTP packets (RFC 3984)
- `GstUDPSink` - sends UDP packets to the network. When paired with RTP payloader (`Gst-rtph264pay`) it can implement RTP streaming

<a name='01_ex_base'></a>
## 1.1.3 Exercise: Build and Run the Base Application

In the `deepstream-test1` example, object detection is performed on a per-frame-basis. Counts for `Vehicle` and `Person` objects are also tracked.  Bounding boxes are drawn around the objects identified, and a counter display is overlayed in the upper left corner of the video. 

#### Build the DeepStream app
Execute the following cell to build the application:
- Click on the cell to select it
- Press [SHIFT][ENTER] or [CONTROL][ENTER] on your keyboard to execute the instructions in the code cell.  Alternatively, you can click the run button at the top of the notebook.


```python
# Build the app
%cd /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out
!make clean
!make
```

#### Run the DeepStream app
Open the VLC media player on your laptop:
- Click "Media" and open the  "Open Network Stream" dialog
- Set the URL to `rtsp://192.168.55.1:8554/ds-test`
- Start execution of the cell below
- Click "Play" on your VLC media player right after you start executing the cell.  

The stream will start shortly from the Jetson Nano and display in the media player.  If you find you've missed it due to a time out in the media player, try the process again, this time waiting a little longer before starting the media player.


```python
# Run the app
%cd /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out
!./deepstream-test1-app /home/dlinano/deepstream_sdk_v4.0.2_jetson/samples/streams/sample_720p.h264
```

<a name='01_change_objects'></a>
# 1.2 Configure an Object Detection Model

The sample application shows counts for two types of objects: `Vehicle` and `Person`.  However, the model that is used can actually detect four types of objects, as revealed in the application C code (line 46):

```c
gchar pgie_classes_str[4][32] = { "Vehicle", "TwoWheeler", "Person",
  "Roadsign"
};
```


<a name='01_config'></a>
## 1.2.1 `Gst-nvinfer` Configuration File
This information is specific to the model used for the inference, which in this case is a sample model provided with the DeepStream SDK.  The `Gst-nvinfer` plugin employs a configuration file to specify the model and various properties. Open the configuration file for the app we are using at [/home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out/dstest1_pgie_config.txt](../deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out/dstest1_pgie_config.txt).  The `Gst-nvinfer` configuration file uses a “Key File” format, with details on key names found in the DeepStream Plugin Manual (use the link provided in the class pages for more details). 
- The **\[property\]** group configures the general behavior of the plugin. It is the only mandatory group.
- The **\[class-attrs-all\]** group configures detection parameters for all classes.
- The **\[class-attrs-\<class-id\>\]** group configures detection parameters for a class specified by \<class-id\>. For example, the \[class-attrs-2\] group configures detection parameters for class ID 2\. This type of group has the same keys as \[class-attrs-all\]. 

Note that the number of classes and the ordered `labels.txt` file are specified in the \[property\] group along with the model engine. For this exercise, we are more interested in configuring the \[class-attrs-all\] and \[class-attrs-\<class-id\>\] groups.  In the sample, we see the following:

```c
[class-attrs-all]
threshold=0.2
eps=0.2
group-threshold=1
```

The `threshold=0.2` key sets the detection confidence score. This tells us that all objects with a 20% confidence score or better will be marked as detected. If the threshold were greater than 1.0, then no objects could ever be detected.  

This "all" grouping is not granular enough if we only want to detect a subset of the objects possible, or if we want to use a different confidence level with different objects.  For example, we might want to detect only vehicles, or we might want to identify people with a different confidence level than road signs.  To specify a threshold for the four individual objects available in this model, add a specific group to the config file for each class: 
* \[class-attrs-0\] for vehicles
- \[class-attrs-1\] for bicycles
- \[class-attrs-2\] for persons
- \[class-attrs-3\] for road signs

In each group, we can now specify the threshold value.  This will be used to determine object detection for each of the four object categories individually.

<a name='01_ex_change'></a>
## 1.2.2 Exercise: Detect Only Two Object Types
Create a new app based on `deepstream-test1-rtsp_out` that detects **only** cars and bicycles.  Start by copying the existing app to a new workspace.


```python
# Create a new app located at /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-two-objects 
#      based on deepstream-test1-rtsp_out
%cd /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps
!mkdir -p my_apps/dst1-two-objects
!cp -rfv dli_apps/deepstream-test1-rtsp_out/* my_apps/dst1-two-objects
```

Using what you just learned, modify the [configuration file](../deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-two-objects/dstest1_pgie_config.txt) in your new app to only detect cars and bicycles.  You will need to add *class-specific groups* for each of the four classes to the end of your configuration file.<br>
Class-specific example:
   ```
   # Per class configuration
   # car
   [class-attrs-0] 
   threshold=0.2
   ```
Then, build and run the app to see if it worked!


```python
# Build the app
%cd /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-two-objects
!make clean
!make
```


```python
# Run the app
%cd /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-two-objects
!./deepstream-test1-app /home/dlinano/deepstream_sdk_v4.0.2_jetson/samples/streams/sample_720p.h264
```

#### How did you do?
If you see something like this image, you did it!  If not, keep trying or take a peek at the solution code in the solutions directory. If you aren't satisfied with the detection of the bicycle, you can experiment with the confidence threshold value. <br>

<img src="images/01_bikes_and_cars.png">

<a name='01_count_objects'></a>
# 1.3 Modify Metadata to Perform Analysis

The object detection is working well, but we are only counting the `Person` and `Vehicle` objects detected.  We would like to show the counts for the bicycles instead of people.  The `Gst-nvinfer` plugin finds objects and provides metadata about them as an output on its source pad, which is passed along through the pipeline.   Using a GStreamer **probe**, we can take a look at the metadata and count the objects detected downstream.  This extraction of the information occurs at the input, or "sink pad", of the `Gst-nvdsosd` plugin.

<img src="images/01_test1_app.png" >

<a name='01_probe'></a>
## 1.3.1 Extracting Metadata with a GStreamer Probe
The `osd_sink_pad_buffer_probe` code in [the deepstream-test1 app](../deepstream_sdk_v4.0.2_jetson/sources/apps/dli_apps/deepstream-test1-rtsp_out/deepstream_test1_app.c) is a callback that is run each time there is new frame data. With this probe, we can snapshot the metadata coming into the `Gst-nvdsosd` plugin, and count the current objects.  The metadata collected that we want to look at will be collected in `obj_meta`: 

```c
NvDsObjectMeta *obj_meta = NULL;
```

The `NvDsObjectMeta` data structure includes an element for the `class_id`.  This is the same class number used in the config file to identify object types: 
* 0 for vehicles
* 1 for bicycles
* 2 for persons
* 3 for road signs

The _for_ loop in the probe checks the `obj_meta->class_id` value for every object in the frame and counts them as needed.

```c
#define PGIE_CLASS_ID_VEHICLE 0
#define PGIE_CLASS_ID_PERSON 2

...

static GstPadProbeReturn
osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info,
    gpointer u_data)
{
    GstBuffer *buf = (GstBuffer *) info->data;
    guint num_rects = 0; 
    NvDsObjectMeta *obj_meta = NULL;
    guint vehicle_count = 0;
    guint person_count = 0;
    NvDsMetaList * l_frame = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsDisplayMeta *display_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next) {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
        int offset = 0;
        for (l_obj = frame_meta->obj_meta_list; l_obj != NULL;
                l_obj = l_obj->next) {
            obj_meta = (NvDsObjectMeta *) (l_obj->data);
            if (obj_meta->class_id == PGIE_CLASS_ID_VEHICLE) {
                vehicle_count++;
                num_rects++;
            }
            if (obj_meta->class_id == PGIE_CLASS_ID_PERSON) {
                person_count++;
                num_rects++;
            }
        }
...

```

The count for each is then added to a display buffer, which is then added to the frame metadata.

```c
...
    
        display_meta = nvds_acquire_display_meta_from_pool(batch_meta);
        NvOSD_TextParams *txt_params  = &display_meta->text_params[0];
        display_meta->num_labels = 1;
        txt_params->display_text = g_malloc0 (MAX_DISPLAY_LEN);
        offset = snprintf(txt_params->display_text, MAX_DISPLAY_LEN, "Person = %d ", person_count);
        offset = snprintf(txt_params->display_text + offset , MAX_DISPLAY_LEN, "Vehicle = %d ", vehicle_count);

...
    
        nvds_add_display_meta_to_frame(frame_meta, display_meta);
    }
      
...

```

In summary, there are four places that require changes if we want to modify the counts:
* Constants for the class ID values (similar to `PGIE_CLASS_ID_VEHICLE`)
* Variables to track the counts (similar to `vehicle_count`
* _if_ statements to check the objects and count them
* `snprintf` statements to fill the buffer for displaying the counts

<a name='01_ex_count'></a>
## 1.3.2 Exercise: Count Vehicles and Bikes
Create a new app based on `deepstream-test1-rtsp_out` that shows counts for vehicles and bicycles.  Fill in the following cells with appropriate commands to create, build, and run your app. To edit your files, use the JupyterLab file browser at left to navigate to the correct folder; then, double click on the file you wish to open and edit.


```python
# TODO
# Create a new app located at /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-counts 
#     based on deepstream-test1-rtsp_out
```


```python
# TODO
# Edit the C-code to count vehicles and bicycles
# Build the app
```


```python
# TODO
# Run the app
```

#### How did you do?
If you see something like this image, you did it!  If not, keep trying or take a peek at the solution code in the solutions directory. You can also modify the `g_print` lines to provide bicycle count feedback while the stream is running. 

<img src="images/01_counts.png">

<a name='01_final'></a>
# 1.4 Putting It All Together

Great job!  You've learned how to build a pipeline, detect various objects, and probe/modify the metadata to count the objects.  It's time to put what you've learned about objects and metadata into one new app.

<a name='01_ex_challenge'></a>
## 1.4.1 Exercise: Detect and Count Three Object Types
Create a new app based on `deepstream-test1-rtsp_out` that detects and shows counts for only three kinds of objects: persons, vehicles, and bicycles.  Adjust the confidence values if needed for each.  Fill in the following cells with appropriate commands to create, build, and run your app.  


```python
# TODO
# Create a new app located at /home/dlinano/deepstream_sdk_v4.0.2_jetson/sources/apps/my_apps/dst1-three-things
```


```python
# TODO
# Edit the C-code to include counts for Persons, Vehicles, and Bikes
# Hint: For the offset in the display, you will need to account for two different offsets to properly place the third value.
# Build the app
```


```python
# TODO
# Run the app
```

#### How did you do?
If you see something like this image, you did it!  If not, keep trying or take a peek at the solution code in the solutions directory.

<img src="images/01_three_things.png">

<h2 style="color:green;">Congratulations!</h2>

You've created DeepStream apps to detect and count objects in a scene in various configurations.<br>
Move on to [2.0 Multiple Networks Application](./02_MultipleNetworks.ipynb).

<center><img src="images/DLI Header.png" alt="Header" width="400"></center>
