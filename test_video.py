from model import MyCNN
import torch
import cv2
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--data-path', type=str, default='./natural_scenes/video')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--model-path', type=str, default='Save_Model/best.pt')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyCNN(num_classes=6)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # DATA
    categories = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    video_path = args.data_path + '/NatureSceneShort.mp4'
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter("Visualization/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    
    # INFERENCE
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        while cap.isOpened():
            flag, frame = cap.read()
            if not flag: break  
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.image_size, args.image_size))
            image = np.transpose(image, (2, 0, 1)) / 255.0
            image = np.expand_dims(image, axis=0)
            image = torch.from_numpy(image).float()
            image = image.to(device)

            prediction = model(image)
            prob = softmax(prediction)
            max_value, max_index = torch.max(prob, dim=1)
            cv2.putText(frame, f'{categories[max_index]}: {max_value[0]:.4f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)
    cap.release()
    out.release()
            
    