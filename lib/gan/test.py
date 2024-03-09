
import sys
sys.path.append('../../')

import os
import torch
import argparse
import numpy as np

from PIL import Image
from torchvision.utils import make_grid

from lib.gan.modules import Generator
from lib.gan.dataset import EMNISTDataModule


def find_best_model(directory):
    """Find the best model in the given directory based on validation loss."""
    best_model_path = None
    best_val_loss = float('inf')

    for filename in os.listdir(directory):
        if filename.endswith('.ckpt'):
            checkpoint_path = os.path.join(directory, filename)
            checkpoint = torch.load(checkpoint_path)
            val_loss = checkpoint['val_loss']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_path

    return best_model_path


def parse_input_file(file_path):
    """Parse the input file and return a list of EMNIST class indices."""
    with open(file_path, 'r') as file:
        content = file.read().strip()

    datamodule = EMNISTDataModule()
    class_to_idx = {cls: idx for idx, cls in enumerate(datamodule.classes)}

    class_indices = [class_to_idx[char] for char in content]
    return class_indices


def generate_images(generator, class_indices, batch_size, device):
    """Generate images using the generator model."""
    generated_images = []
    numeric_results = []

    for i in range(0, len(class_indices), batch_size):
        batch_indices = class_indices[i:i+batch_size]
        batch_labels = torch.tensor(batch_indices, dtype=torch.long).to(device)

        z = torch.randn(len(batch_indices), generator.latent_dim).to(device)
        batch_images = generator(z, batch_labels).detach().cpu()

        generated_images.append(batch_images)
        numeric_results.append(batch_labels.cpu())

    generated_images = torch.cat(generated_images)
    numeric_results = torch.cat(numeric_results)
    return generated_images, numeric_results


def save_results(generated_images, numeric_results, output_dir):
    """Save the generated images and numeric results."""
    image_grid = make_grid(generated_images, nrow=10)
    image_grid = (image_grid + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    image_grid = (image_grid * 255).numpy().astype(np.uint8)
    image_grid = np.transpose(image_grid, (1, 2, 0))
    image = Image.fromarray(image_grid)

    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, 'generated_images.png')
    image.save(image_path)

    numeric_results_path = os.path.join(output_dir, 'numeric_results.pt')
    torch.save(numeric_results, numeric_results_path)


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
        latent_dim=checkpoint['latent_dim'], num_classes=checkpoint['num_classes'])
    generator.load_state_dict(checkpoint['state_dict'])
    generator.to(device)
    generator.eval()

    # Parse the input file
    class_indices = parse_input_file(input_file)

    # Generate images
    generated_images, numeric_results = generate_images(
        generator, class_indices, batch_size, device)

    # Save the results
    save_results(generated_images, numeric_results, output_dir)

    print('Image generation completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate images using a trained model.')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the directory containing the trained models.')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to the input file containing the class indices.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory.')
    args = parser.parse_args()

    main(args)
