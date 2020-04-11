# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import pandas as pd
import cv2
from darkflow.net.build import TFNet
import os
from moviepy.editor import VideoFileClip
import schedule
from datetime import date,datetime,timedelta
import subprocess
import schedule
import time
import glob
from dateutil import parser
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import math

os.environ['TZ'] = 'Asia/Kolkata'
time.tzset()

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


def front_face_detector(videoFile):
    option = {
    'model': 'cfg/yolov2-2c.cfg',
    'load': 1625,
    'threshold': 0.1,
    'gpu': 0.95
}
    tfnet = TFNet(option)
    capture = cv2.VideoCapture(videoFile)
    totalFrames = 0
    fps = capture.get(5)
    fps_round = round(fps)
    filename = videoFile.split('/')[1]
    df = pd.read_csv('hourly_csv_op/'+filename.split('.')[0]+'.csv')
    age_proto = "deploy_age.prototxt"
    age_model = "age_net.caffemodel"
    age_net = cv2.dnn.readNet(age_model, age_proto)
    age_list = ['(0 - 2)', '(4 - 6)', '(8 - 12)', '(15 - 20)', '(25 - 32)', '(38 - 43)', '(48 - 53)', '(60 - 100)']
    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    col_men_face_secs = []
    col_women_face_secs = []
    malefacesec = 0
    femalefacesec = 0
    sec_count = 0
    col_age_0_21_count = []
    col_age_21_35_count = []
    col_age_35_count = []
    age_group1 = 0
    age_group2 = 0
    age_group3 = 0
    duration = int(60)
    colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
    gender_model = load_model('gender_face.h5')
    gender_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    genders = ['female','male']
    out = cv2.VideoWriter('05-07-19_09-00facesec.avi', get_video_type(filename), fps, (int(capture.get(3)),int(capture.get(4))))

    while (capture.isOpened()):
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            results = tfnet.return_predict(frame)
            for color, result in zip(colors, results):
                tl = (result['topleft']['x'], result['topleft']['y'])
                br = (result['bottomright']['x'], result['bottomright']['y'])
                face = frame[tl[1]-15:br[1]+15, tl[0]-15:br[0]+15]
                age = 0
                try:
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=True)
                    age_net.setInput(blob)
                    age_pred = age_net.forward()
                    age = age_list[age_pred[0].argmax()]
                    print('age',age)
                    if age_pred[0].argmax() < 4:
                        age_group1+=1
                        print('1 +1')
                    elif age_pred[0].argmax() == 4:
                        age_group2+=1
                        print('2 +1')
                    elif age_pred[0].argmax() > 4:
                        age_group3+=1
                        print('3 +1')
                except:
                    pass
                label = 'front_face'
                frame = cv2.rectangle(frame, tl, br, color, 7)
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                im = Image.fromarray(frame, 'RGB')
                im = im.resize((64,64))
                img_array = np.array(im)
                img_array = np.expand_dims(img_array, axis=0)
                result = gender_model.predict(img_array)
                gender = genders[int(result[0][0])]
                if gender == 'male':
                    malefacesec+=1
                    print('male')
                elif gender == 'female':
                    femalefacesec+=1
                    print('female')        

            if totalFrames % fps_round == 0:
                    sec_count+=1
                    if sec_count % 60 == 0:
                        col_men_face_secs.append(math.ceil(malefacesec/fps_round))
                        col_women_face_secs.append(math.ceil(femalefacesec/fps_round))
                        col_age_0_21_count.append(round(age_group1/(fps_round*60)))
                        col_age_21_35_count.append(round(age_group2/(fps_round*60)))
                        col_age_35_count.append(round(age_group3/(fps_round*60)))
                        age_group1 = 0
                        age_group2 = 0
                        age_group3 = 0
                        malefacesec = 0
                        femalefacesec = 0


            # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
            out.write(frame)
            totalFrames+=1
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            capture.release()
            cv2.destroyAllWindows()
            break

    #Ensure face sec lists length are of 60
    if len(col_men_face_secs) != duration:
        if len(col_men_face_secs) < duration:
            for i in range(len(col_men_face_secs),duration):
                col_men_face_secs.append('N/A')
                col_women_face_secs.append('N/A')
        else:
            col_men_face_secs[duration-1] = sum(col_men_face_secs[duration-1:])
            col_men_face_secs = col_men_face_secs[:duration]
            col_women_face_secs[duration-1] = sum(col_women_face_secs[duration-1:])
            col_women_face_secs = col_women_face_secs[:duration]


    #Ensure age lists length are of 60
    if len(col_age_0_21_count) != duration:
        if len(col_age_0_21_count) < duration:
            for i in range(len(col_age_0_21_count),duration):
                col_age_0_21_count.append(0)
                col_age_21_35_count.append(0)
                col_age_35_count.append(0)
        else:
            col_age_0_21_count[duration-1] = sum(col_age_0_21_count[duration-1:])
            col_age_0_21_count = col_age_0_21_count[:duration]
            col_age_21_35_count[duration-1] = sum(col_age_21_35_count[duration-1:])
            col_age_21_35_count = col_age_21_35_count[:duration]
            col_age_35_count[duration-1] = sum(col_age_35_count[duration-1:])
            col_age_35_count = col_age_35_count[:duration]

    df['age_0_21_count'] = col_age_0_21_count
    df['age_21_35_count'] = col_age_21_35_count
    df['age_35+count'] = col_age_35_count
    df['men_face_secs'] = col_men_face_secs
    df['women_face_secs'] = col_women_face_secs
    # print(col_men_face_secs)
    # print(col_women_face_secs)

    df.to_csv('hourly_csv_op/'+filename.split('.')[0]+'.csv',index=False)

def track_gender(videoFile):
    options = {
    'model': 'cfg/tiny-yolo-voc-2c.cfg',
    'load': 12375,
    'threshold': 0.1,
    'gpu': 0.95
}

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--skip-frames", type=int, default=5,
    help="# of skip frames between detections")
    args = vars(ap.parse_args())



    filename = videoFile.split('/')[1]
    col_date = filename.split('.')[0].split('_')[0]
    col_date = time.strptime(col_date, "%d-%m-%y")
    col_date = time.strftime("%m/%d/%Y",col_date)
    weekday = parser.parse(col_date).strftime("%a")
    col_time = filename.split('.')[0].split('_')[1]
    col_time = time.strptime(col_time, "%H-%M")
    col_time = time.strftime("%H:%M:%S",col_time)
    timestamps = pd.timedelta_range(col_time, periods=60, freq="1T")
    col_date_time = []
    col_ts = []
    col_start_hour = []
    col_minute = []
    col_total_men = []
    col_total_women = []
    col_unique_men = []
    col_unique_women = []
    for i in range(len(timestamps)):
        col_start_hour.append(str(timestamps[i]).split(' ')[2].split(':')[0])
        col_minute.append(str(timestamps[i]).split(' ')[2].split(':')[1])
        col_ts.append(str(timestamps[i]).split(' ')[2].split(':')[0] + ':' + str(timestamps[i]).split(' ')[2].split(':')[1])
        col_date_time.append(col_date + ' '+ col_ts[i])

    # clip = VideoFileClip(videoFile)
    # duration = clip.duration
    # clip.close()
    duration = int(60)

    df = pd.DataFrame({'date_time': col_date_time,'date':col_date,'weekday':weekday,'start_hour':col_start_hour,'minute':col_minute,'time':col_ts})



    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    maxDisappeared = 40
    mct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=50)
    fct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=50)
    maletrackers = []
    maletrackableObjects = {}
    femaletrackers = []
    femaletrackableObjects = {}
    maleFrames = []
    femaleFrames = []
    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalMales = 0
    totalFemales = 0
    prevMales = 0
    prevFemales = 0
    totalmalecounter = 0
    totalmaleseconds = 0
    totalfemalecounter = 0
    totalfemaleseconds = 0
    newMales = []
    newFemales = []

    tfnet = TFNet(options)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]
    capture = cv2.VideoCapture(videoFile)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    fps = capture.get(5)
    fps_round = round(fps)

    out = cv2.VideoWriter('05-07-19_09-00gender.avi', get_video_type(filename), fps, (int(capture.get(3)),int(capture.get(4))))



    print('args["skip_frames"]',args["skip_frames"])


    while True:
            # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        stime = time.time()
        ret, frame = capture.read()
        if ret:
            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib

            # frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]



            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            malerects = []
            femalerects = []


            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % args["skip_frames"] == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                maletrackers = []
                femaletrackers = []


                results = tfnet.return_predict(frame)

                # loop over the detections
                for color, result in zip(colors, results):
                    # extract the confidence (i.e., probability) associated
                    # with the prediction
                    confidence = result['confidence']
                    label = result['label']

                    # if the class label is not a person, ignore it
                    if label.lower() != 'm' and label.lower() != 'f':
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    (startX, startY, endX, endY) = (result['topleft']['x'], result['topleft']['y'],result['bottomright']['x'], result['bottomright']['y'])

                    

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    if label.lower() == 'm':
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        maletracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        maletracker.start_track(rgb, rect)
                        maletrackers.append(maletracker)
                        continue
                    elif label.lower() == 'f':
                        # construct a dlib rectangle object from the bounding
                        # box coordinates and then start the dlib correlation
                        # tracker
                        femaletracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        femaletracker.start_track(rgb, rect)
                        femaletrackers.append(femaletracker)
                        continue

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            else:
                # loop over the trackers
                for tracker in maletrackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    malerects.append((startX, startY, endX, endY))
                
                for tracker in femaletrackers:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # add the bounding box coordinates to the rectangles list
                    femalerects.append((startX, startY, endX, endY))


            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            maleobjects = mct.update(malerects)

            # loop over the tracked objects
            for (objectID, centroid) in maleobjects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = maletrackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)
                    totalmalecounter+=1
                    

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if totalFrames % fps_round == 0:
                        totalmaleseconds+=1
                        if totalmaleseconds % 60 == 0:
                            col_total_men.append(totalmalecounter)
                            totalmalecounter = 0

                    # check to see if the object has been counted or not
                    if not to.counted:

                        totalMales += 1
                        to.counted = True

                # store the trackable object in our dictionary
                maletrackableObjects[objectID] = to
                maleFrames.append(totalMales)

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "Male ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            
            femaleobjects = fct.update(femalerects)

            # loop over the tracked objects
            for (objectID, centroid) in femaleobjects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = femaletrackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)
                    totalfemalecounter+=1
                    


                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)


                    if totalFrames % fps_round == 0:
                        totalfemaleseconds+=1
                        if totalfemaleseconds % 60 == 0:
                            col_total_women.append(totalfemalecounter)
                            totalfemalecounter = 0

                    # check to see if the object has been counted or not
                    if not to.counted:

                        totalFemales += 1
                        to.counted = True

                # store the trackable object in our dictionary
                femaletrackableObjects[objectID] = to
                femaleFrames.append(totalFemales)

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "Female ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("MaleCount", totalMales),
                ("FemaleCount", totalFemales),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            # if writer is not None:
            #     writer.write(frame)

            # show the output frame
            # cv2.imshow("Frame", frame)
            # key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break


            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1

            if totalFrames % fps_round == 0:
                newMales.append(totalMales - prevMales)
                newFemales.append(totalFemales - prevFemales)
                if len(newMales) % 60 == 0:
                    col_unique_men.append(sum(newMales))
                    newMales = []
                if len(newFemales) % 60 == 0:
                    col_unique_women.append(sum(newFemales))
                    newFemales = []
                prevMales = totalMales
                prevFemales = totalFemales

            out.write(frame)
            # print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        
        else:
            break

    #Ensure total males & females list length is of 60
    if len(col_total_men) != duration:
        if len(col_total_men) < duration:
            for i in range(len(col_total_men),duration):
                col_total_men.append(0)
        else:
            col_total_men[duration-1] = sum(col_total_men[duration-1:])
            col_total_men = col_total_men[:duration]
    
    if len(col_total_women) != duration:
        if len(col_total_women) < duration:
            for i in range(len(col_total_women),duration):
                col_total_women.append(0)
        else:
            col_total_women[duration-1] = sum(col_total_women[duration-1:])
            col_total_women = col_total_women[:duration]    

    #Ensure unique males & females list length is of 60
    if len(col_unique_men) != duration:
        if len(col_unique_men) < duration:
            for i in range(len(col_unique_men),duration):
                col_unique_men.append(0)
        else:
            col_unique_men[duration-1] = sum(col_unique_men[duration-1:])
            col_unique_men = col_unique_men[:duration]

    if len(col_unique_women) != duration:
        if len(col_unique_women) < duration:
            for i in range(len(col_unique_women),duration):
                col_unique_women.append(0)
        else:
            col_unique_women[duration-1] = sum(col_unique_women[duration-1:])
            col_unique_women = col_unique_women[:duration]


    df['total_men'] = col_total_men
    df['total_women'] = col_total_women
    df['Unique_men'] = col_unique_men
    df['Unique_women'] = col_unique_women

    df.to_csv('hourly_csv_op/'+filename.split('.')[0]+'.csv',index=False)


    capture.release()
    cv2.destroyAllWindows()
def hourly_to_daily_to_final():
    os.chdir("hourly_csv_op/")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    filename = ''
    if len(all_filenames)>0:
        filename = all_filenames[0].split('_')[0]
        #combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
        #export to csv
        os.chdir("..")
        combined_csv.to_csv("daily_csv_op/"+filename+".csv", index=False, encoding='utf-8-sig')
        daily = pd.read_csv("daily_csv_op/"+filename+".csv")
        final = pd.read_csv('final_csv_op/final_data.csv')
        combined_csv = pd.concat([final, daily])
        combined_csv.to_csv('final_csv_op/final_data.csv', index=False, encoding='utf-8-sig')
        os.system('gdrive upload --parent 1sL_0Lv2AWrdLO2bYUVHuDBI497a4_nXB final_csv_op/final_data.csv')


def download_videos():
    #download
    command = "gsutil ls -l gs://overall-data/videos_from_screen/*.avi | sort -rk 2| awk '{print $3 " " $4}'"
    op = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
    videos =  str(op.communicate()[0].decode("utf-8"))
    videos = videos.split('\n')
    videos = [video for video in videos if 'gs://' in video]
    from datetime import date,datetime,timedelta
    d = date.today() - timedelta(days=1)
    dt = datetime.strftime(d, "%d-%m-%y")
    recent_videos = []
    for i in range(len(videos)):
        if dt in videos[i]:
            recent_videos.append(videos[i].split('gs://overall-data/videos_from_screen/')[1])
            get = 'gsutil cp '+videos[i]+' raw_videos/'
            # op = subprocess.Popen(get,shell=True,stdout=subprocess.DEVNULL)
            os.system(get)
    
    recent_videos.sort()
    print(recent_videos)
    # recent_videos = ['05-07-19_09-00.avi']

    #fps change for each video
    for video in recent_videos:
        video_path = 'raw_videos/'+ video 
        save_filename = 'preprocessed_videos/'+ video
        cap = cv2.VideoCapture(video_path)
        frames_per_second = float(cap.get(7)/3600)
        print('fps',frames_per_second)
        out = cv2.VideoWriter(save_filename, get_video_type(save_filename), frames_per_second, (int(cap.get(3)),int(cap.get(4))))
        while True:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else: 
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        delete_video = 'rm '+video_path
        os.system(delete_video)

    #run models
    for video in recent_videos:
        video_path = 'preprocessed_videos/'+ video
        while True:
            f = open("labels.txt", "w")
            f.write("f\nm")
            f.close()
            track_gender(videoFile = video_path)
            f = open("labels.txt", "w")
            f.write("male_face\nfemale_face")
            f.close()
            front_face_detector(videoFile = video_path)
            break
        # delete_video = 'rm '+video_path
        # os.system(delete_video)
    hourly_to_daily_to_final()

def delete_from_bucket():
    command = "gsutil ls -l gs://overall-data/videos_from_screen/*.avi | sort -rk 2| awk '{print $3 " " $4}'"
    op = subprocess.Popen(command,shell=True,stdout=subprocess.PIPE)
    videos =  str(op.communicate()[0].decode("utf-8"))
    videos = videos.split('\n')
    videos = [video for video in videos if 'gs://' in video]
    from datetime import date,datetime,timedelta
    d = date.today() - timedelta(days=3)
    dt = datetime.strftime(d, "%d-%m-%y")
    dt = time.strptime(dt, "%d-%m-%y")
    deleted_videos = []
    for i in range(len(videos)):
        new_d = videos[i].split('gs://overall-data/videos_from_screen/')[1].split('_')[0]
        newdate = time.strptime(new_d, "%d-%m-%y")
        if newdate < dt:
            deleted_videos.append(videos[i])
            delete = 'gsutil rm '+videos[i]
            os.system(delete)

if __name__ == "__main__":

    schedule.every().day.at("18:30").do(delete_from_bucket)
    schedule.every().day.at("20:30").do(download_videos)

 
    while True: 
        # Checks whether a scheduled task 
        # is pending to run or not 
        schedule.run_pending() 
        time.sleep(1)
    # download_videos()
    # track_gender(videoFile = 'preprocessed_videos/05-07-19_09-00.avi')
    # front_face_detector(videoFile = 'preprocessed_videos/05-07-19_09-00.avi')