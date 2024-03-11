
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader


def display_samples(args):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the EMNIST dataset
    dataset = EMNIST(root=args.data_path, split='balanced',
                     train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Create a dictionary to store the displayed classes
    displayed_classes = {}

    # Idx to class mapping
    idx_to_class = {}

    # Iterate over each sample in the dataset
    for images, labels in dataloader:
        # Get the class label
        label = dataset.classes[labels.item()]

        # Check if the class has already been displayed
        if label not in displayed_classes:
            # Display the sample and label
            image = images.squeeze().numpy()
            plt.imshow(image, cmap='gray')
            plt.title(f"Class: {label}")
            plt.axis('off')
            plt.show()

            # Update the idx to class mapping
            idx_to_class[labels.item()] = label

            # Mark the class as displayed
            displayed_classes[label] = True
    
        if len(displayed_classes) == 62 or len(idx_to_class) == 62:
            break

    # Save the idx to class mapping to a JSON file
    with open(os.path.join(args.out_dir, 'idx_to_class.json'), 'w') as f:
        json.dump(idx_to_class, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='EMNIST Test')
    parser.add_argument('--data-path', type=str,
                        default='../data', help='Path to the EMNIST dataset')
    parser.add_argument('--out-dir', type=str,
                        default='../data', help='Path to the output directory')
    args = parser.parse_args()

    display_samples(args)


if __name__ == '__main__':
    main()
