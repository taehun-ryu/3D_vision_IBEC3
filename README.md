# IBEC$^3$: IWE-Based Event Camera Calibration with Checkerboard
## Get started
### 1. Environment
We recommend using a conda virtual environment.
```
conda env create -f config/environment.yaml
```

### 2. Data conversion
We provide a data conversion file so that you can easily convert it to the data structure of the h5 file we use.
Supported data types are: `.bag`, `.db3`, `.aedat4`

To do this, bellow dependencies is required:
- rpg_dvs_ros
- dv-processing-python

For ROS data, Run `python3 data_format/rosbag_to_h5.py /path/to/your/bagfile --output_dir /path/to/your/outputdir --event_topic /dvs/events --zero_ts`. (ROS message type only supports [dvs_msgs](https://github.com/uzh-rpg/rpg_dvs_ros/tree/master/dvs_msgs))

Example:
```bash
# For `.bag` file,
python3 data_format/rosbag_to_h5.py /root/dataset/checkerboard.bag --output_dir /root/dataset/h5/ --event_topic /dvs/events --zero_ts
# For `.db3` file,
python3 data_format/db3_to_h5.py /root/dataset/checkerboard/ --output_dir /root/dataset/h5/ --event_topic /dvs/events --zero_ts
```

The .aedat4 file format is used when recording data with software provided by [iniVation](https://docs.inivation.com/software/dv/gui/record-playback.html). We provide a reliable conversion script for data recorded using the DAVIS346 model.
```bash
python3 data_format/aedat_to_h5.py /root/dataset/checkerboard.aedat4 --output_dir /root/dataset/h5/ --output_name checker.h5 --zero_ts
```

### 3. Run
Edit [config/calibration.yaml](https://github.com/taehun-ryu/3D_vision_IBEC3/blob/main/config/calibration.yaml) to match your dataset and calibration setup.
```yaml
path: "/path/to/events.h5"         # Path to input HDF5 event file

img_size: [260, 346]               # Image size [height, width]

board_w: 4                         # Number of inner corners along the checkerboard width
board_h: 4                         # Number of inner corners along the checkerboard height

square_size: 4.0                   # Size of one checker square (in mm or other real-world units)

user_selecting: false             # If true, manually select corners; otherwise, auto-detect

visualization:                    # Visualization options
  iwe: false                      # Show IWE (Image of Warped Events)
  corner: false                   # Show detected checkerboard corners
  calib: true                     # Show calibration result (e.g., reprojection)

```
Run calibration code
```bash
python3 main.py
```