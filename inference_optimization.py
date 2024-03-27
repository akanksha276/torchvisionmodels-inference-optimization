import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import json
import matplotlib.pyplot as plt

def load_model(model_name):
    if model_name == 'inception':
        return models.inception_v3(pretrained=True)
    elif model_name == 'resnet':
        return models.resnet50(pretrained=True)
    elif model_name == 'vgg':
        return models.vgg16(pretrained=True)

def preprocess_image(image_path, model_type):
    '''
    different pre-processing as per different models
    '''
    if model_type == 'inception':
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

def run_inference(model, input_tensor):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output

def class_id_to_label(i):
    '''
    function to return ImageNet class label from input id
    '''
    # Get the predicted class label name
    with open('imagenet-simple-labels.json') as f:
        labels = json.load(f)
    return labels[i]

def main(args):
    model = load_model(args.model)
    input_tensor = preprocess_image(args.image,args.model)

    if args.optimization == 'none':
        start_time = time.time()
        output = run_inference(model, input_tensor)
        end_time = time.time()
        predicted_class = torch.argmax(output, dim=1).item()
        print(f'\n{args.model}:')
        print(f'Execution Time (Before Optimization): {end_time - start_time:.4f} seconds')
        print(f'Predicted Class Index and Label (Without Optimization): {predicted_class}',': ',class_id_to_label(predicted_class))

    elif args.optimization == 'low_resolution':
        start_time = time.time()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        output = run_inference(quantized_model, input_tensor)
        end_time = time.time()
        predicted_class = torch.argmax(output, dim=1).item()
        print(f'\n{args.model}:')
        print(f'Execution Time (With Optimization - Low Resolution): {end_time - start_time:.4f} seconds')
        print(f'Predicted Class Index and Label (With Optimization - Low Resolution): {predicted_class}',': ',class_id_to_label(predicted_class))

    elif args.optimization == 'torchscript':
        start_time = time.time()
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        with torch.no_grad():
            scripted_output = scripted_model(input_tensor)
        end_time = time.time()

        # Extract logits from the scripted output as per model type
        if args.model == 'inception':
          logits = scripted_output.logits
        else:
          logits = scripted_output
          
        # Compute the predicted class index from the logits
        predicted_class = torch.argmax(logits, dim=1).item()
        print(f'\n{args.model}:')
        print(f'Execution Time (With Optimization - TorchScript): {end_time - start_time:.4f} seconds')
        print(f'Predicted Class Index and Label (With Optimization - TorchScript): {predicted_class}',': ',class_id_to_label(predicted_class))

    else:
        print("Unsupported optimization option.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with different optimizations.")
    parser.add_argument("--model", type=str, choices=['inception', 'resnet', 'vgg'], default='inception',
                        help="Choose the model for inference.")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image.")
    parser.add_argument("--optimization", type=str, choices=['none', 'low_resolution', 'torchscript'], default='none',
                        help="Choose the optimization method.")
    args = parser.parse_args()
    
    main(args)