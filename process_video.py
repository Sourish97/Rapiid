import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
import os

option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 72816,
    'threshold': 0.3,
    'gpu': 0.95
}
# Video Encoding, might require additional installs
# Types of Codes: http://www.fourcc.org/codecs.php
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    #'mp4': cv2.VideoWriter_fourcc(*'H264'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

tfnet = TFNet(option)
capture = cv2.VideoCapture('TestVideo_test_video.mp4')
filename = 'output.avi'
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
out = cv2.VideoWriter(filename, get_video_type(filename), capture.get(5), (int(capture.get(3)),int(capture.get(4))))

while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        out.write(frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break

#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#ffmpeg_extract_subclip("client-60mins.avi", 270, 290, targetname="client.avi")