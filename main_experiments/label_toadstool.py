import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from models import *
import json
import numpy as np
import os
import torch
from torchvision import transforms
from helpers import *
from deepface import DeepFace

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


classes = [
    'anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'neutral'
]

trfm = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize((0.5313),(0.2110))     
    ])

trfm2 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))   
    ])

grayscale = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),     
    ])

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])


def get_frames(video,length=None):
    frames = []
    v_cap = cv2.VideoCapture(video)
    if length == None:
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 30
    else:
        v_len = length
   

    last_frame = None
    last_face = None

    for fn in range(v_len):  # Drop last second as video length is 2101 seconds
        success, frame = v_cap.read()
        if success:
            last_frame = frame
        elif not success:
            frame = last_frame
        
        if (fn % 10 == 0): #(fn % 30 == 0) or ((fn+1) % 30 == 0):
            
            face = fdc.detect_face(frame)
           
            src = cv2.imread("ck_reference.jpg")
              
            if len(face) == 0:
                print("Did not detect face, using previous frame")
                face = last_face
            else:
                last_face = face
            
            matched = hist_match(face,src)
            
            data = np.asarray(matched).astype(
                np.uint8)
            frames.append(data)
        

    v_cap.release()
    return frames


def predict_seq(frames,model,seq_len,step_size,trfm,device,participant_nr):
    predictions = {}
    frame_indices = [2,2,3,3]
    for start in range(0,len(frames),step_size):
        jump_index = 0
        seq = []
        if (start+seq_len)>len(frames):
            break
        for index in frame_indices:
            for frame in frames[int(start+jump_index):int(start+jump_index+index)]:
                seq.append(trfm(frame))
            jump_index += 3
        with torch.no_grad():
            model.eval()
            out = model(torch.stack(seq,0).permute(1,0,2,3).unsqueeze(0).cuda())
            _, pred = torch.max(out, dim=1)
            soft = F.softmax(out,dim=1).squeeze(0)
            label = classes[pred]
            prob = soft[pred.item()].item() * 100
            prob_dis = [p.item() * 100 for p in soft]
            seq_title = str(int(start/3))+"-"+str(int(start/3+4)-1)
            predictions[seq_title] = {"Emotion Label": label, "Probability score":prob, "Probability distributions":prob_dis}
            
        
    
    with open(participant_nr + "_prediction4sec_unprocessed.json", 'w') as fp:
        json.dump(predictions, fp)


def get_video_values(frames,grayscale):
    gray_frames = [grayscale(frame) for frame in frames]

    mean = torch.mean(torch.stack(gray_frames,0))
    std = torch.std(torch.stack(gray_frames,0))

    print(mean,std)

def relabel_majority(jsonfile,p):
    with open(jsonfile, 'r') as fp:
        preds = json.load(fp)
        labels = [v['Emotion Label'] for v in preds.values()]
        most_common = max(labels, key=labels.count)
        if most_common != 'neutral':
            for pred in preds.items():
                
                if (pred[1]['Emotion Label'] == most_common) and pred[1]['Probability score'] < 90:
                    preds[pred[0]]['Emotion Label'] = 'neutral'
        
    with open("participant_"+str(p)+"_prediction.json", 'w') as fp:
        json.dump(preds, fp)            
        
def label_sequences():

    device = get_default_device()
    model = C3D(sample_size=112,
                 sample_duration=10,
                 num_classes=7)

    to_device(model,device)

    state1 = torch.load("trained_model_weights/C3D_224_Adam_weights0.pt")
    state2 = torch.load("trained_model_weights/C3D_224_Adam_weights1.pt")
    state3 = torch.load("trained_model_weights/C3D_224_Adam_weights2.pt")
    for key in state1:
         state1[key] = (state1[key] + state2[key] + state3[key]) / 3.

    model.load_state_dict(state1)

    for i in range(10):

        participant_nr = "participant_" + str(i)
        seq_len = 10
        step_size = 3

        if not os.path.exists(participant_nr+'_3FramesPerSec'+'.npy'):
            print("Extracting frames for participant: " + participant_nr)
            frames = get_frames(participant_nr+"_video.avi")
            frames = np.asarray(frames)
            np.save(participant_nr+'_3FramesPerSec'+'.npy', frames, allow_pickle=True)
        else:
            frames = np.load(participant_nr+'_3FramesPerSec'+'.npy', allow_pickle=True)
        predict_seq(frames,model,seq_len,step_size,grayscale,device,participant_nr)

def label_individual_images():
    model = ResNet(BasicBlock, [2, 2, 2, 2], 7)

    model.load_state_dict(torch.load(
        'trained_model_weights/model_parameter_resCK.pt', map_location=torch.device('cpu')))


    participant_nr = "participant_9"
    folder_predictions = {}
    for img in range(0, 2100):
        folder_path = participant_nr+"/images/"

        path = folder_path + str(img) + ".jpg"
        res = DeepFace.analyze(img_path=path,
                            actions=["emotion"], enforce_detection=False)
        with torch.no_grad():

            model.eval()
            transformed = transform(cv2.imread(path))
            out = model(transformed.unsqueeze(0))
            soft = F.softmax(out[0])
            labelindex = torch.argmax(soft).item()
            label = classes[labelindex]
            prob = soft[labelindex].item() * 100

        dominant_emotion = res["dominant_emotion"]
        pred_score = res["emotion"][dominant_emotion]
        folder_predictions[img] = {"DeepFace": [
            dominant_emotion, pred_score], "ResNet": [label, prob]}
        print("Predicted for image: " + path)
    with open(folder_path + "prediction.json", 'w') as fp:
        json.dump(folder_predictions, fp)