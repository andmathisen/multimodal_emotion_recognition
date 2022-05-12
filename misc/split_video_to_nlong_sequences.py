import cv2
import os
import numpy as np
import face_detection_cropping as fdc


def get_frames(filename, n_frames=30):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fn in range(v_len-30):  # Drop last second as video length is 2101 seconds
        success, frame = v_cap.read()
        if success is False:
            print("fail")
            continue
        if (fn % 30 == 0):
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    print(len(frames))
    v_cap.release()
    return frames, v_len


def save_faces_in_sequences(frames, parent_dir):
    seq_nr = 0
    count = 1

    for idx, frame in enumerate(frames):
        # print(idx)
        if idx % 5 == 0:
            path = os.path.join(parent_dir, str(seq_nr))
            if os.path.exists(path) == False:
                os.mkdir(path)
            count = 1
            seq_nr = seq_nr + 5

        face = fdc.detect_face(frame)
        if len(face) == 0:  # Unable to detect face in frame
            print("Could not detect face, skipped frame")
            face = prev_face  # If unable to find face use last found face
            cv2.imwrite((path+'/' + str(count) + ".jpg"), face)
            print("Saved face image to: " + path+'/' + str(count) + ".jpg")
        else:
            prev_face = face
            cv2.imwrite((path+'/' + str(count) + ".jpg"), face)
            print("Saved face image to: " + path+'/' + str(count) + ".jpg")
        count = count + 1


def save_faces(frames, path):
    for idx, frame in enumerate(frames):
        face = fdc.detect_face(frame)
        if len(face) == 0:  # Unable to detect face in frame
            print("Could not detect face, skipped frame")
            face = prev_face  # If unable to find face use last found face
            cv2.imwrite((path+'/' + str(idx) + ".jpg"), face)
            print("Saved face image to: " + path+'/' + str(idx) + ".jpg")
        else:
            prev_face = face
            cv2.imwrite((path+'/' + str(idx) + ".jpg"), face)
            print("Saved face image to: " + path+'/' + str(idx) + ".jpg")


participant_nr = "participant_9"

frames, v_len = get_frames(
    "/Users/andreas/Desktop/master/toadstool/participants/"+participant_nr+"/"+participant_nr+"_video.avi")

# save_faces_in_sequences(
#     frames, "/Users/andreas/Desktop/master/toadstool/participants/"+participant_nr+"/img_sequences")

save_faces(frames, "/Users/andreas/Desktop/master/toadstool/participants/" +
           participant_nr+"/images")
