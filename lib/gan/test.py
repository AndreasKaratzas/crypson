
import sys
sys.path.append('../../')

import os
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision.utils import make_grid

from lib.gan.modules import Generator


def find_best_model(directory):
    """Find the best model in the given directory 
    based on validation loss. Validation loss is 
    given in the checkpoint filename. Checkpoints 
    are saved as 'epoch_{epoch}-loss_{val_loss}.ckpt'.
    
    Parameters
    ----------
    directory : str
        Path to the directory containing the trained models.

    Returns
    -------
    str
        Path to the best model.

    Raises
    ------
    ValueError
        If no model is found in the given directory.
    """
    best_model_path = None
    best_val_loss = float('inf')

    for filename in os.listdir(directory):
        if filename.endswith('.ckpt'):
            val_loss = float(filename.split(
                '-')[1].split('_')[1].split('.')[0])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(directory, filename)

    if best_model_path is None:
        raise ValueError('No model found in the given directory.')

    return best_model_path


def parse_input_file(file_path):
    # TODO: Utilize the deterministic file under the data directory (`idx_to_class.json`)
    """Parse the input file and return a list of EMNIST class indices."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Define the EMNIST class labels
    emnist_classes = [str(i) for i in range(10)] + [chr(i) for i in range(
        ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)] + [' ']

    class_to_idx = {cls: idx for idx, cls in enumerate(emnist_classes)}
    class_indices = []

    for line in lines:
        line = line.strip()
        for char in line:
            if char in class_to_idx:
                class_indices.append(class_to_idx[char])
            else:
                class_indices.append(class_to_idx[' '])
        class_indices.append(class_to_idx[' '])  # Add a space between lines

    return class_indices


def generate_images(generator, class_indices, device, img_size=32):
    """Generate images using the generator model."""
    generated_images = []
    space_index = class_indices[-1]  # Index of the space character

    batch_labels = torch.tensor(
        [class_idx for class_idx in class_indices if class_idx != space_index]).to(device)
    z = torch.randn(len(batch_labels), generator.latent_dim).to(device)
    batch_images = generator(z, batch_labels).detach(
    ).cpu().view(-1, 1, img_size, img_size)
    space_images = torch.zeros(class_indices.count(
        space_index), 1, img_size, img_size)

    # Merge generated images and space images
    merged_images = []
    space_count = 0
    for class_idx in class_indices:
        if class_idx == space_index:
            merged_images.append(space_images[space_count])
            space_count += 1
        else:
            merged_images.append(batch_images[class_idx - space_count])

    merged_images = torch.stack(merged_images)
    generated_images.append(merged_images)
    generated_images = torch.cat(generated_images)

    return generated_images


def save_results(generated_images, output_dir):
    """Save the generated images and numeric results."""
    image_grid = make_grid(generated_images, nrow=10)
    image_grid = (image_grid + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    image_grid = (image_grid * 255).numpy().astype(np.uint8)
    image_grid = np.transpose(image_grid, (1, 2, 0))
    image = Image.fromarray(image_grid)

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, 'generated_images.png')
    image.save(image_path)


def main(args):
    # Set the path to the directory containing the trained models
    model_dir = args.model_dir

    # Set the path to the input file
    input_file = args.input_file

    # Set the output directory
    output_dir = args.output_dir

    # Set the batch size for image generation
    batch_size = 128

    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the best model
    best_model_path = find_best_model(model_dir)

    # Load the best model
    checkpoint = torch.load(best_model_path)
    generator = Generator(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        img_size=args.img_size,
        hidden_dim=args.hidden_dim)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    generator.eval()

    # Parse the input file
    class_indices = parse_input_file(input_file)

    # Generate images
    generated_images = generate_images(generator, class_indices,
                                       device, img_size=args.img_size)

    # Save the results
    save_results(generated_images, output_dir)

    print('Image generation completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate images using a trained model.')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the directory containing the trained models.')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input file containing the class indices.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory.')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Dimension of the latent space.')
    parser.add_argument('--num-classes', type=int, default=62,
                        help='Number of classes in the dataset.')
    parser.add_argument('--img-size', type=int, default=32,
                        help='Size of the input images.')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Dimension of the hidden layer in the generator model.')
    args = parser.parse_args()

    main(args)
