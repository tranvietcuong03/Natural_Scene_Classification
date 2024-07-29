from model import MyCNN
import torch
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--data-path', type=str, default='./natural_scenes/test')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--model-path', type=str, default='Save_Model/best.pt')
    return parser.parse_args()

def inference(origin_images):
    softmax = torch.nn.Softmax()
    images = []
    predictions = []
    probabilities = []
    for img in origin_images:
        image = cv2.resize(img, (args.image_size, args.image_size))
        image = np.transpose(image, (2, 0, 1)) / 255.0  # (3, 224, 224)
        image = np.expand_dims(image, axis=0)  
        image = torch.from_numpy(image).float()  # torch.Size([1, 3, 224, 224])
        image = image.to(device)

        with torch.no_grad():
            prediction = model(image)
            probability = softmax(prediction)
        max_value, max_index = torch.max(probability, dim=1)

        images.append(img)
        predictions.append(categories[max_index])
        probabilities.append(max_value[0])

    visulize(images, predictions, probabilities)

def visulize(images, predictions, probabilities):
    fig, axes = plt.subplots(2,4, figsize=(12,8))
    for i, (image, pred, prob) in enumerate(zip(images, predictions, probabilities)):
        ax = axes[i // 4, i % 4]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(cv2.resize(image, (300,300)))
        ax.set_title(f'{pred}: {prob:.4f}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

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
    list_images = ['5.jpg', '6.jpg', '22.jpg', '28.jpg', '51.jpg', '93.jpg', '101.jpg', '121.jpg']
    all_image_paths = [args.data_path + '/' + image for image in list_images]
    origin_images = [cv2.imread(image_path) for image_path in all_image_paths]
    
    # INFERENCE
    inference(origin_images)
    
