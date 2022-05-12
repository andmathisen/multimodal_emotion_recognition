import cv2
import sys

import torch.nn.functional as F
from torchvision import transforms
import face_detection_cropping as fdc
import resnet18_ck as res
import FER_model as fer
import torch
import numpy as np
from PIL import Image
from models import *

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

classes = [
    'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral', 'contempt'
]
classes_fer = ['angry', 'disgust', 'fear',
               'happy', 'sad', 'surprise', 'neutral']


class predictor:
    vid = None

    def predict_segment(self, video, model, transform):
        vidcap = cv2.VideoCapture(video)

        success, image = vidcap.read()
        tot_prob = 0
        cur_label = None
        start_t = 0
        end_t = 0
        framecount = 0
        imagecount = 0
        sequence_len = 0
        with open('predicted_sequences' + '_' + participant_nr + '.txt', 'w') as f:
            while success:

                if framecount % 30 == 0:
                    face = fdc.detect_face(image)
                    if len(face) == 0:
                        print("skippedframe")
                        pass
                    else:
                        model.eval()
                        transformed = transform(face)
                        
                        out = model(transformed.unsqueeze(0))

                        soft = F.softmax(out[0])
                        labelindex = torch.argmax(soft).item()
                        label = classes[labelindex]
                        prob = soft[labelindex].item()
                        print(label)
                        print("Probability:", prob)
                        if label != cur_label:
                            if cur_label == None:

                                start_t = imagecount
                                end_t = imagecount
                            else:
                                end_t = imagecount-1

                                cv2.imwrite(
                                    participant_nr+'/'+label+'/'+str(start_t)+'_'+str(prob)+'.jpg', face)
                                avg_prob = tot_prob/sequence_len
                                f.write(cur_label + ', ' + "{:.2f}".format(avg_prob*100) + '%, Start time: ' + str(
                                    start_t) + ' End time: ' + str(end_t) + ', Sequence length:'+str(sequence_len)+'\n')
                                start_t = imagecount
                                tot_prob = 0
                                sequence_len = 0
                            cur_label = label
                        sequence_len += 1
                        tot_prob += prob
                    imagecount += 1

                success, image = vidcap.read()
                framecount += 1

                #print(str(framecount) + "----" + str(imagecount))
if __name__ == "__main__":

    predictor = predictor()
    participant_nr = "participant_6"
    model = ResNet(BasicBlock, [2, 2, 2, 2], 7)
    model.load_state_dict(torch.load('trained_model_weights/model_parameter_resCK.pt',map_location=torch.device('cpu')))
    path = "/Users/andreas/Desktop/master/toadstool/participants/"
    
    full_path = path + participant_nr+ "/" + participant_nr+'_video.avi'
    # model.eval()
    predictor.predict_segment(full_path, model, transform)
    """
        img = cv2.imread('test.jpg')
        face = fdc.detect_face(img)

        transformed = transform(face)
        out = model(transformed.unsqueeze(0))
        print(F.softmax(out[0]))
        labelindex = torch.argmax(out[0]).item()
        print(classes[labelindex])
        """
