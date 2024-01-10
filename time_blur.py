import torch
import time
from torchvision import transforms, datasets
from TinyEyesTransformOOPcuda import TinyEyes
import os
from PIL import Image
from torchvision.utils import save_image

def main():
    # Timing different ways of calculating a random matrix and multiplying
    # Only Method A accelerates well on GPU as others all have CPU bottleneck
    traindir =  '/data/ILSVRC2012/val_in_folders'

    # Define instance of TinyEyes class
    for age in ['week0','week4','week8','week12']:
        print(f'age {age}')
        TinyEyes_withParams = TinyEyes(age=age, width=15.8, dist=60, imp='gpu')

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(
            traindir,
            train_transforms
            )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True,
            num_workers=16, sampler=None)

        model = torch.nn.Sequential(
            TinyEyes_withParams,  # Add TinyEyes Transform to Model for GPU Implementation
        )
        
        data_time = []
        end = time.time()

        nits = 4



        model.cuda()

        os.makedirs('example_images', exist_ok=True)
        print(len(train_loader))
        for batch_idx, (images, target) in enumerate(train_loader):
            if batch_idx>=nits:
                break
            images = images.cuda('cuda', non_blocking=True)
            output = model(images)

            save_image(output, f'example_images/age-{age}_batch-{batch_idx}_example.jpg')
            
            # measure data loading time
            data_time.append(time.time() - end)
            print(f' batch {batch_idx} time {data_time[-1]}')
            end = time.time()

        print(data_time)

if __name__ == "__main__":
    main()