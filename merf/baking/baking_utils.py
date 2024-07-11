import functools
import gc
import itertools

import numpy as np
import scipy.ndimage
import tqdm
from PIL import Image
import multiprocessing
import json

def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def parallel_write_images(image_write_fn, img_and_path_list):
    """Parallelizes image writing over CPU cores with progress bar.

    Args:
    - image_write_fn: A function that takes a tuple as input (path, image) and
        writes the result to disk.
    - img_and_path_list: A list of tuples (image, path) containing all the images
        that should be written.
    """
    
    # multiprocessing 
    with multiprocessing.Pool() as pool:
        for _ in tqdm.tqdm(pool.imap(image_write_fn, img_and_path_list), total=len(img_and_path_list)):
            pass
          
def save_8bit_png(img_and_path):
  """Save an 8bit numpy array as a PNG on disk.

  Args:
    img_and_path: A tuple of an image (numpy array, 8bit, [height, width,
      channels]) and a path where the image is saved (string).
  """
  img, pth = img_and_path
  with open_file(pth, 'wb') as imgout:
    Image.fromarray(img).save(imgout, 'PNG')


def as_mib(x):
  """Computes size of array in Mebibyte (MiB)."""
  return x.size / (1024**2)

def save_json(x, pth):
  with open_file(pth, 'w') as f:
    json.dump(x, f)

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0.0, 1.0) * 255.0).astype(np.uint8)
    ).save(f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')



def get_atlas_block_size(data_block_size):
  """Add 1-voxel apron for native trilerp in the WebGL renderer."""
  return data_block_size + 1


def reshape_into_3d_atlas_and_compute_indirection_grid(
    sparse_grid_features_1d,
    sparse_grid_density_1d,
    data_block_size,
    alive_macroblocks,
):
  """Reshapes into 3D atlas and computes indirection grid."""
  atlas_block_size = get_atlas_block_size(data_block_size)
  num_occupied_blocks = sparse_grid_features_1d.shape[0]

  # Find 3D texture dimensions with lowest storage impact with a brute
  # force search.
  def compute_az(ax, ay):
    num_blocks_per_atlas_unit = ax * ay
    az = int(np.ceil(num_occupied_blocks / num_blocks_per_atlas_unit))
    return az, ax * ay * az

  best_num_occupied_blocks_padded = np.inf
  for ax_cand, ay_cand in itertools.product(range(1, 255), range(1, 255)):
    az, num_occupied_blocks_padded = compute_az(ax_cand, ay_cand)

    # Make sure that the volume texture does not exceed 2048^3 since
    # some devices do not support abitrarily large textures (although the limit
    # is usally higher than 2048^3 for modern devices). Also make sure that
    # resulting indices will be smaller than 255 since we are using a single
    # byte to encode the indices and the value 255 is reserved for empty blocks.
    if (
        num_occupied_blocks_padded < best_num_occupied_blocks_padded
        and az < 255
        and ax_cand * atlas_block_size <= 2048
        and ay_cand * atlas_block_size <= 2048
        and az * atlas_block_size <= 2048
    ):
      ax, ay = ax_cand, ay_cand
      best_num_occupied_blocks_padded = num_occupied_blocks_padded
  az, num_occupied_blocks_padded = compute_az(ax, ay)

  # Make sure that last dim is smallest. During .png export we slice this volume
  # along the last dim and sorting ensures that not too many .pngs are created.
  ax, ay, az = sorted([ax, ay, az], reverse=True)

  # Add padding, if necessary.
  required_padding = num_occupied_blocks_padded - num_occupied_blocks
  if required_padding > 0:

    def add_padding(x):
      padding = np.zeros((required_padding,) + x.shape[1:])
      return np.concatenate([x, padding], axis=0)

    sparse_grid_features_1d = add_padding(sparse_grid_features_1d)
    sparse_grid_density_1d = add_padding(sparse_grid_density_1d)

  # Reshape into 3D texture.
  def reshape_into_3d_texture(x):
    x = x.reshape(
        ax, ay, az,
        atlas_block_size, atlas_block_size, atlas_block_size, x.shape[-1])
    x = x.swapaxes(2, 3).swapaxes(1, 2).swapaxes(3, 4)
    return x.reshape(
        ax * atlas_block_size, ay * atlas_block_size, az * atlas_block_size,
        x.shape[-1],
    )

  sparse_grid_features = reshape_into_3d_texture(sparse_grid_features_1d)
  sparse_grid_density = reshape_into_3d_texture(sparse_grid_density_1d)

  # Compute indirection grid.
  block_indices_compact = np.arange(num_occupied_blocks)
  block_indices_compact = np.unravel_index(block_indices_compact, [ax, ay, az])
  block_indices_compact = np.stack(block_indices_compact, axis=-1)
  index_grid_size = alive_macroblocks.shape
  sparse_grid_block_indices = -1 * np.ones(
      (index_grid_size[0], index_grid_size[1], index_grid_size[2], 3), np.int16
  )
  sparse_grid_block_indices[alive_macroblocks] = block_indices_compact

  return sparse_grid_features, sparse_grid_density, sparse_grid_block_indices
