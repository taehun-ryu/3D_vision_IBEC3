import glob
import argparse
import rosbag
import rospy
from cv_bridge import CvBridge, CvBridgeError
import os
import h5py
import numpy as np
from event_packagers import *
from tqdm import tqdm


def append_to_dataset(dataset, data):
    dataset.resize(dataset.shape[0] + len(data), axis=0)
    if len(data) == 0:
        return
    dataset[-len(data):] = data[:]


def timestamp_float(ts):
    return ts.secs + ts.nsecs / float(1e9)


def get_rosbag_stats(bag, event_topic, image_topic=None, flow_topic=None):
    num_event_msgs = 0
    num_img_msgs = 0
    num_flow_msgs = 0
    topics = bag.get_type_and_topic_info().topics
    for topic_name, topic_info in topics.items():
        if topic_name == event_topic:
            num_event_msgs = topic_info.message_count
            print('Found events topic: {} with {} messages'.format(topic_name, topic_info.message_count))
        if topic_name == image_topic:
            num_img_msgs = topic_info.message_count
            print('Found image topic: {} with {} messages'.format(topic_name, num_img_msgs))
        if topic_name == flow_topic:
            num_flow_msgs = topic_info.message_count
            print('Found flow topic: {} with {} messages'.format(topic_name, num_flow_msgs))
    return num_event_msgs, num_img_msgs, num_flow_msgs


# Inspired by https://github.com/uzh-rpg/rpg_e2vid 
# Modified for Python 3 
# Fixed zero_ts implementation
def extract_rosbag(rosbag_path, output_path, event_topic, image_topic=None,
                   flow_topic=None, start_time=None, end_time=None, zero_ts=False,
                   packager=hdf5_packager, is_color=False):
    ep = packager(output_path)
    first_ts = None
    t0 = None
    sensor_size = None

    if not os.path.exists(rosbag_path):
        print(f"{rosbag_path} does not exist!")
        return

    with rosbag.Bag(rosbag_path, 'r') as bag:
        num_event_msgs, num_img_msgs, num_flow_msgs = get_rosbag_stats(
            bag, event_topic, image_topic, flow_topic)

        xs, ys, ts, ps = [], [], [], []
        max_buffer_size = 1e20
        ep.set_data_available(num_img_msgs, num_flow_msgs)
        num_pos = num_neg = img_cnt = flow_cnt = 0

        for topic, msg, _ in tqdm(bag.read_messages()):
            if first_ts is None:
                if topic == event_topic and msg.events:
                    raw_t0 = timestamp_float(msg.events[0].ts)
                else:
                    raw_t0 = timestamp_float(msg.header.stamp)
                first_ts = raw_t0
                t0 = 0.0 if zero_ts else raw_t0

            # --- EVENTS ---
            if topic == event_topic:
                for e in msg.events:
                    raw_t = timestamp_float(e.ts)
                    rel_t = raw_t - (first_ts if zero_ts else 0.0)

                    xs.append(e.x)
                    ys.append(e.y)
                    ts.append(rel_t)
                    ps.append(1 if e.polarity else 0)

                    if e.polarity:
                        num_pos += 1
                    else:
                        num_neg += 1

                    last_ts = rel_t

                # flush if buffer too big
                if len(xs) > max_buffer_size:
                    if sensor_size is None:
                        sensor_size = [max(ys) + 1, max(xs) + 1]
                    ep.package_events(xs, ys, ts, ps)
                    xs.clear(); ys.clear(); ts.clear(); ps.clear()

            # --- IMAGES (optional) ---
            elif topic == image_topic:
                raw_t = timestamp_float(msg.header.stamp)
                rel_t = raw_t - (first_ts if zero_ts else 0.0)

                if is_color:
                    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
                else:
                    image = CvBridge().imgmsg_to_cv2(msg, "mono8")

                ep.package_image(image, rel_t, img_cnt)
                sensor_size = image.shape
                img_cnt += 1

            # --- FLOW (optional) ---
            elif topic == flow_topic:
                raw_t = timestamp_float(msg.header.stamp)
                rel_t = raw_t - (first_ts if zero_ts else 0.0)

                flow_x = np.array(msg.flow_x).reshape(msg.height, msg.width)
                flow_y = np.array(msg.flow_y).reshape(msg.height, msg.width)
                flow_image = np.stack((flow_x, flow_y), axis=0)

                ep.package_flow(flow_image, rel_t, flow_cnt)
                flow_cnt += 1

        # final flush of any remaining event buffer
        if xs:
            if sensor_size is None:
                sensor_size = [max(ys) + 1, max(xs) + 1]
            ep.package_events(xs, ys, ts, ps)

        if sensor_size is None:
            raise Exception("ERROR: No sensor size detected, implies no events/images in bag topics?")

        # metadata: num_pos, num_neg, duration, t0, last_ts, img_cnt, flow_cnt, sensor_size
        ep.add_metadata(
            num_pos,
            num_neg,
            last_ts - t0,
            t0,
            last_ts,
            img_cnt,
            flow_cnt,
            sensor_size
        )


def extract_rosbags(rosbag_paths, output_dir, event_topic, image_topic, flow_topic,
        zero_ts=False, is_color=False):
    for path in rosbag_paths:
        bagname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(bagname))
        print("Extracting {} to {}".format(path, out_path))
        extract_rosbag(path, out_path, event_topic, image_topic=image_topic,
                       flow_topic=flow_topic, zero_ts=zero_ts, is_color=is_color)


if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="ROS bag file to extract or directory containing bags")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Folder where to extract the data (default: same folder as each .bag)"
    )
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument("--image_topic", default=None, help="Image topic (if left empty, no images will be collected)")
    parser.add_argument("--flow_topic", default=None, help="Flow topic (if left empty, no flow will be collected)")
    parser.add_argument('--zero_ts', action='store_true', help='If true, timestamps will be offset to start at 0')
    parser.add_argument('--is_color', action='store_true', help='Set flag to save frames from image_topic as 3-channel, bgr color images')
    args = parser.parse_args()

    # Gather all .bag paths
    if os.path.isdir(args.path):
        rosbag_paths = sorted(glob.glob(os.path.join(args.path, "*.bag")))
    else:
        rosbag_paths = [args.path]

    # Process each bag individually
    for bag in rosbag_paths:
        name = os.path.splitext(os.path.basename(bag))[0]

        if args.output_dir:
            out_dir = args.output_dir
        else:
            bag_folder = os.path.dirname(bag)
            out_dir = os.path.dirname(bag_folder)

        os.makedirs(out_dir, exist_ok=True)
        print(f"Data will be extracted in folder: {out_dir}")
        extract_rosbags(
            [bag],
            out_dir,
            args.event_topic,
            args.image_topic,
            args.flow_topic,
            zero_ts=args.zero_ts,
            is_color=args.is_color
        )
