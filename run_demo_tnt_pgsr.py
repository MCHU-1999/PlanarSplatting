import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'planarsplat'))
import argparse
import torch
import numpy as np
from pyhocon import ConfigFactory
from pyhocon import ConfigTree
from utils_demo.run_planarSplatting import run_planarSplatting
from planarsplat.data_process.scannetpp.colmap_io import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from PIL import Image
import cv2


def get_depth_normal(depth_prior_path: str, normal_prior_path: str, img_id_list):
    normal_maps_list = []
    depth_maps_list = []

    for img_id in img_id_list:
        # Format the integer ID into a 6-digit string with leading zeros
        filename = f"{img_id:06d}.npy"

        # Load depth map
        depth_file_path = os.path.join(depth_prior_path, filename)
        if os.path.exists(depth_file_path):
            depth_map = np.load(depth_file_path)
            depth_maps_list.append(depth_map)
        else:
            raise FileNotFoundError(f"Depth file not found: {depth_file_path}")

        # Load normal map
        normal_file_path = os.path.join(normal_prior_path, filename)
        if os.path.exists(normal_file_path):
            normal_map = np.load(normal_file_path)
            normal_maps_list.append(normal_map)
        else:
            raise FileNotFoundError(f"Normal file not found: {normal_file_path}")

    return depth_maps_list, normal_maps_list

def get_process_depth_normal(depth_prior_path: str, normal_prior_path: str, img_id_list):
    normal_maps_list = []
    depth_maps_list = []

    for img_id in img_id_list:
        # Format the integer ID into a 6-digit string with leading zeros
        filename = f"{img_id:06d}.npy"

        # Load and process depth map
        depth_file_path = os.path.join(depth_prior_path, filename)
        if os.path.exists(depth_file_path):
            depth_map = np.load(depth_file_path)
            depth_map = np.clip(depth_map, 0, 300)  # Clip depth to [0, 300]
            depth_maps_list.append(depth_map)
        else:
            raise FileNotFoundError(f"Depth file not found: {depth_file_path}")

        # Load and process normal map
        normal_file_path = os.path.join(normal_prior_path, filename)
        if os.path.exists(normal_file_path):
            normal_map = np.load(normal_file_path)
            normal_map = (normal_map + 1) / 2.0  # Normalize normals to [0, 1]
            normal_maps_list.append(normal_map)
        else:
            raise FileNotFoundError(f"Normal file not found: {normal_file_path}")

    return depth_maps_list, normal_maps_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-d", "--data_path", type=str, default='path/to/colmap/data', help='path of input colmap data')
    parser.add_argument("-g", "--geo_data_path", type=str, default='parent/of/normal_and_depth/data', help='path of parent directory of rendered depth and normal maps')
    parser.add_argument("-o", "--out_path", type=str, default='planarSplat_ExpRes/demo_tnt_pgsr', help='path of output dir')
    parser.add_argument("--conf_path", type=str, default='utils_demo/demo_tnt_pgsr.conf', help='path of configure file')
    parser.add_argument('--use_precomputed_data', default=False, action="store_true", help='use processed data from input images')
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f'The input data path {data_path} does not exist.')
    
    geo_data_path = args.geo_data_path
    if not os.path.exists(geo_data_path):
        raise ValueError(f'The input geometric priors data path {geo_data_path} does not exist.')
    else:
        depth_prior_path = os.path.join(geo_data_path, "renders_depth")
        normal_prior_path = os.path.join(geo_data_path, "renders_normal")
    
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        raise ValueError(f'The input image path {image_path} does not exist.')

    colmap_cam_file_path = os.path.join(data_path, 'sparse/cameras.bin')
    if not os.path.exists(colmap_cam_file_path):
        raise ValueError(f'The input path {colmap_cam_file_path} does not exist.')
    
    colmap_image_file_path = os.path.join(data_path, 'sparse/images.bin')
    if not os.path.exists(colmap_image_file_path):
        raise ValueError(f'The input path {colmap_image_file_path} does not exist.')


    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    precomputed_data_path = os.path.join(out_path, 'data.pth')
    use_precomputed_data = args.use_precomputed_data

    if use_precomputed_data and os.path.exists(precomputed_data_path):
        data = torch.load(precomputed_data_path)
        print(f"loading precomputed data from {precomputed_data_path}")
    else:
        cameras = read_intrinsics_binary(colmap_cam_file_path)
        camera = next(iter(cameras.values()))
        fx, fy, cx, cy = camera.params[:4]
        intrinsic = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.0]]).astype(np.float32)
        h = camera.height
        w = camera.width
        
        images_meta = read_extrinsics_binary(colmap_image_file_path)
        
        color_images_list = []
        image_paths_list = []
        c2ws_list = []
        intrinsics_list = []
        img_id_list = []

        i = 0
        for img_id, img_meta in images_meta.items():
            frame_name = img_meta.name
            frame_path = os.path.join(image_path, frame_name)

            q = img_meta.qvec
            t = img_meta.tvec
            r = qvec2rotmat(q)
            rt = np.eye(4)
            rt[:3,:3] = r
            rt[:3, 3] = t
            c2w = np.linalg.inv(rt).astype(np.float32)
            rgb = np.array(Image.open(frame_path))  # h, w, 3

            c2ws_list.append(c2w)
            intrinsics_list.append(intrinsic)
            image_paths_list.append(frame_path)
            color_images_list.append(rgb)
            img_id_list.append(img_id)

        # Fetch pre-computed, aligned depth and normal maps
        depth_maps_list, normal_maps_list = get_depth_normal(depth_prior_path, normal_prior_path, img_id_list)

        data = {
            'color': color_images_list,
            'depth': depth_maps_list,
            'normal': normal_maps_list,
            'image_paths': image_paths_list,
            'extrinsics': c2ws_list,  # c2w
            'intrinsics': intrinsics_list,
            'out_path': out_path
        }
        torch.save(data, precomputed_data_path)

    # load conf
    base_conf = ConfigFactory.parse_file('planarsplat/confs/base_conf_planarSplatCuda.conf')
    demo_conf = ConfigFactory.parse_file(args.conf_path)
    conf = ConfigTree.merge_configs(base_conf, demo_conf)
    conf.put('train.exps_folder_name', out_path)
    img_res = [data['color'][0].shape[0], data['color'][0].shape[1]]
    conf.put('dataset.img_res', img_res)
    conf.put('dataset.pre_align', True)
    conf.put('dataset.voxel_length', 0.1)
    conf.put('dataset.sdf_trunc', 0.2)
    conf.put('plane_model.init_plane_num', 3000)

    planar_rec = run_planarSplatting(data=data, conf=conf)