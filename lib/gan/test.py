
import sys
sys.path.append('../../')

import os
import json
import torch
import argparse
import numpy as np

from PIL import Image
from copy import deepcopy
from rich import print as rprint
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
            val_loss = float('.'.join(filename.split(
                '-')[1].split('_')[1].split('.')[:2]))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(directory, filename)

    if best_model_path is None:
        raise ValueError('No model found in the given directory.')

    return best_model_path


def parse_input_file(prompt_path, classes_path):
    """Parse the input prompt file and return 
    a list of EMNIST class indices. The class
    indices are based on the mapping in the
    classes JSON file.
    
    Parameters
    ----------
    prompt_path : str
        Path to the input prompt file.
    classes_path : str
        Path to the classes JSON file.
        
    Returns
    -------
    list
        List of EMNIST class indices.
        
    Raises
    ------
    FileNotFoundError
        If the input prompt file or classes JSON file is not found.
    """
    with open(prompt_path, 'r') as file:
        lines = file.readlines()
    
    with open(classes_path, 'r') as file:
        class_to_idx = json.load(file)

    class_to_idx[' '] = -1
    class_to_idx['\n'] = -2
    tokens = []
    for line in lines:
        for char in line:
            tokens.append(class_to_idx[char])
    
    rprint(f'Prompt: {lines}')
    rprint(f'Class indices: {tokens}')
    return tokens


def generate_images(generator, class_indices, device, 
                    img_size=32, output_dir='../../data', latent_dim=256, 
                    classes_path='../data/idx_to_class.json'):
    """Generate images using the generator model.
    
    Parameters
    ----------
    generator : torch.nn.Module
        Generator model.
    class_indices : list
        List of class indices.
    device : torch.device
        Device (CPU or GPU) to use for image generation.
    img_size : int, optional
        Size of the input images, by default 32.
    output_dir : str, optional
        Path to the output directory, by default '../../data'.
    latent_dim : int, optional
        Dimension of the latent space, by default 256.
    classes_path : str, optional
        Path to the classes JSON file, by default '../data/idx_to_class.json'.
    
    Returns
    -------
    torch.Tensor
        Generated images.
    """
    generated_images = []
    space_index = -1  # Index of the space character
    newline_index = -2  # Index of the newline character

    batch_labels = torch.tensor(
        [class_idx for class_idx in class_indices if (class_idx != space_index and class_idx != newline_index)]).to(device)
    rprint(f'Batch labels: {batch_labels}')
    z = torch.randn(len(batch_labels), latent_dim).to(device)
    rprint(f"Input noise shape: {z.shape}\n Batch labels shape: {batch_labels.shape}")
    batch_images = generator(z, batch_labels).detach(
    ).cpu().view(-1, 1, img_size, img_size)
    space_image = torch.zeros(1, img_size, img_size)
    newline_image = torch.ones(1, img_size, img_size) * (-2)
    # Transpose the images to the correct format
    batch_images = batch_images.transpose(2, 3)

    with open(classes_path, 'r') as file:
        class_to_idx = json.load(file)
    
    class_to_idx[' '] = -1
    class_to_idx['\n'] = -2
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Save the generated images
    for i in range(len(batch_images)):
        label = idx_to_class[batch_labels[i].item()]
        img = batch_images[i]
        img = (img + 1) / 2
        img = (img * 255).squeeze().numpy().astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        img.save(os.path.join(output_dir, f'{label}_{i}.png'))

    # Merge generated images and space images
    merged_images = []
    curr_count = 0
    for class_idx in class_indices:
        if class_idx == space_index:
            merged_images.append(deepcopy(space_image))
        elif class_idx == newline_index:
            merged_images.append(deepcopy(newline_image))
        else:
            merged_images.append(batch_images[curr_count])
            curr_count += 1

    merged_images = torch.stack(merged_images)
    generated_images.append(merged_images)
    generated_images = torch.cat(generated_images)
    
    return generated_images


def save_results(generated_images, output_dir):
    """Save the generated images to the output directory.
    
    Parameters
    ----------
    generated_images : torch.Tensor
        Generated images.
    output_dir : str
        Path to the output directory.
    """
    # Save the generated images
    grid = make_grid(generated_images, nrow=10, normalize=True).permute(1, 2, 0)
    grid = grid.mul(255).clamp(0, 255).to(torch.uint8)
    grid = grid.cpu().numpy()
    grid = Image.fromarray(grid, mode='RGB')
    parent_dir = os.path.dirname(output_dir)
    grid.save(os.path.join(parent_dir, 'generated_images.png'))


def save_results_expanded(generated_images, output_dir):
    # Initialize an empty list to store the image rows
    rows = []
    current_row = []

    for i in range(len(generated_images)):
        img = generated_images[i]
        # Check if the current batch_image contains only -2 values
        if np.allclose(img.numpy(), -2):
            # If the current row is not empty, append it to the rows list
            if current_row:
                rows.append(current_row)
                current_row = []
        else:
            img = (img + 1) / 2
            img = (img * 255).squeeze().numpy().astype(np.uint8)
            img = Image.fromarray(img, mode='L')
            current_row.append(img)

    # If there are any remaining images in the current row, append it to the rows list
    if current_row:
        rows.append(current_row)

    rprint(f'Number of rows: {len(rows)}')
    for i, row in enumerate(rows):
        rprint(f'Row {i}: {len(row)}')

    # Calculate the maximum height of the images in each row
    max_heights = [max(img.height for img in row) for row in rows]

    # Calculate the maximum length of the rows
    max_row_length = max(len(row) for row in rows)

    # Create a new blank image to hold the concatenated rows
    concatenated_image = Image.new(
        'L', (max_row_length * 32, sum(max_heights)))

    # Fill the concatenated image with arrays of 0.5 values
    concatenated_array = np.ones((sum(max_heights), max_row_length * 32)) * 128
    concatenated_image = Image.fromarray(
        concatenated_array.astype(np.uint8), mode='L')

    # Paste the images into the concatenated image
    y_offset = 0
    for row, max_height in zip(rows, max_heights):
        x_offset = 0
        for img in row:
            concatenated_image.paste(img, (x_offset, y_offset))
            x_offset += img.width
        y_offset += max_height

    # Save the concatenated image
    parent_dir = os.path.dirname(output_dir)
    concatenated_image.save(os.path.join(parent_dir, 'generated_images.png'))


def main(args):
    # Set the path to the directory containing the trained models
    model_dir = args.model_dir

    # Set the output directory
    output_dir = args.output_dir

    # Set the device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find the best model
    best_model_path = find_best_model(model_dir)
    rprint(f'Best model found at: {best_model_path}')

    # Load the best model
    checkpoint = torch.load(best_model_path)
    generator = Generator(
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        img_size=args.img_size,)
    generator.load_state_dict(checkpoint['generator'])
    generator.to(device)
    generator.eval()

    # Parse the input file
    class_indices = parse_input_file(args.prompt_path, args.classes_path)

    # Generate images
    generated_images = generate_images(generator, class_indices,
                                       device, img_size=args.img_size,
                                       output_dir=output_dir,
                                       latent_dim=args.latent_dim,
                                       classes_path=args.classes_path)

    # Save the results
    save_results(generated_images, output_dir)

    print('Image generation completed successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate images using a trained model.')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Path to the directory containing the trained models.')
    parser.add_argument('--prompt-path', type=str, required=True,
                        help='Path to the input prompt file.')
    parser.add_argument('--classes-path', type=str, required=True,
                        help='Path to the classes JSON path.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to the output directory.')
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of the latent space.')
    parser.add_argument('--num-classes', type=int, default=47,
                        help='Number of classes in the dataset.')
    parser.add_argument('--img-size', type=int, default=32,
                        help='Size of the input images.')
    args = parser.parse_args()

    main(args)
