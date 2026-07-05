import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect
import cv2
import numpy as np
import math
import os


rospy.init_node('flight')


get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)

bridge = CvBridge()
latest_frame = None


def image_callback(data):
    global latest_frame
    latest_frame = bridge.imgmsg_to_cv2(data, 'bgr8')

image_sub = rospy.Subscriber('main_camera/image_raw', Image, image_callback, queue_size=1)

COLOR_RANGES = {
    'red':     [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    'orange':  [((11, 120, 70), (25, 255, 255))],
    'yellow':  [((26, 120, 70), (35, 255, 255))],
    'green':   [((36, 80, 50), (85, 255, 255))],
    'cyan':    [((86, 80, 50), (100, 255, 255))],
    'blue':    [((101, 80, 50), (130, 255, 255))],
    'purple':  [((131, 60, 50), (155, 255, 255))],
    'pink':    [((156, 60, 70), (169, 255, 255))],
}

COLOR_TO_RGB = {
    'red':    (255, 0, 0),
    'orange': (255, 140, 0),
    'yellow': (255, 255, 0),
    'green':  (0, 255, 0),
    'cyan':   (0, 255, 255),
    'blue':   (0, 0, 255),
    'purple': (128, 0, 128),
    'pink':   (255, 20, 147),
}

WHITE = (255, 255, 255)
MIN_PIXELS = 500


def detect_color(frame):
    if frame is None:
        return None

    h, w = frame.shape[:2]
    cx1, cx2 = int(w * 0.3), int(w * 0.7)
    cy1, cy2 = int(h * 0.3), int(h * 0.7)
    roi = frame[cy1:cy2, cx1:cx2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    best_color = None
    best_count = MIN_PIXELS

    for color_name, ranges in COLOR_RANGES.items():
        mask = None

        for lower, upper in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else (mask | m)

        count = cv2.countNonZero(mask)
        if count > best_count:
            best_count = count
            best_color = color_name

    return best_color


def set_led_color(rgb):
    r, g, b = rgb
    set_effect(r=r, g=g, b=b)


def update_led():
    color_name = detect_color(latest_frame)
    if color_name is not None:
        set_led_color(COLOR_TO_RGB[color_name])
    else:
        set_led_color(WHITE)


def navigate_wait(x=0, y=0, z=1, speed=0.5, frame_id='aruco_map', auto_arm=False, tolerance=0.2):
    navigate(x=x, y=y, z=z, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
            break
        update_led() 
        rospy.sleep(0.2)


def main():
    set_led_color(WHITE)
    navigate_wait(z=1, speed=1, frame_id='body', auto_arm=True)

    waypoints = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
        (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
        (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (1, 3), (0, 3),
        (0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
        (6, 5), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5),
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)
    ]

    for wx, wy in waypoints:
        navigate_wait(x=wx, y=wy, z=1, speed=1, frame_id='aruco_map')
        update_led()
        rospy.sleep(0.3)

    set_led_color(WHITE)
    navigate_wait(x=0, y=0, z=1, speed=1, frame_id='aruco_map')
    land()


if __name__ == "__main__":
    main()