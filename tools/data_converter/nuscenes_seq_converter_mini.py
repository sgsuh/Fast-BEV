# Copyright (c) Phigent Robotics. All rights reserved.
#
# Mini/general variant of nuscenes_seq_converter.py.
#
# The original add_adj_info() is hard-coded to the v1.0-test split (its loop
# skips 'train'/'val' and forces nuscenes_version='v1.0-test'), so it cannot
# build the temporal "4d" info files for the v1.0-mini dataset. This script
# parametrizes the dataset version, the splits to process, and the temporal
# sampling, and writes the same
#   nuscenes_infos_<set>_4d_interval<interval>_max<max_adj>.pkl
# files that the Fast-BEV configs expect.
#
# NOTE: requires numpy < 1.20 because the temporal matching uses np.long.

import argparse
import pickle
from os import path as osp

import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']


def add_adj_info(root_path, version, sets, interval, max_adj):
    nuscenes = NuScenes(version, root_path)

    for split in sets:
        info_path = osp.join(root_path, 'nuscenes_infos_%s.pkl' % split)
        if not osp.exists(info_path):
            print('[skip] %s not found' % info_path)
            continue
        print('=> processing %s (%s)' % (info_path, version))
        dataset = pickle.load(open(info_path, 'rb'))

        map_token_to_id = dict()
        for idx in range(len(dataset['infos'])):
            map_token_to_id[dataset['infos'][idx]['token']] = idx

        for idx in range(len(dataset['infos'])):
            if idx % 10 == 0:
                print('   %d/%d' % (idx, len(dataset['infos'])))
            info = dataset['infos'][idx]
            sample = nuscenes.get('sample', info['token'])

            for adj in ['next', 'prev']:
                sweeps = []
                adj_list = dict()
                for cam in CAMS:
                    adj_list[cam] = []
                    sample_data = nuscenes.get('sample_data', sample['data'][cam])
                    count = 0
                    while count < max_adj:
                        if sample_data[adj] == '':
                            break
                        sd_adj = nuscenes.get('sample_data', sample_data[adj])
                        sample_data = sd_adj
                        adj_list[cam].append(dict(
                            data_path=osp.join(root_path, sd_adj['filename']),
                            timestamp=sd_adj['timestamp'],
                            ego_pose_token=sd_adj['ego_pose_token']))
                        count += 1

                for count in range(interval - 1,
                                    min(max_adj, len(adj_list['CAM_FRONT'])),
                                    interval):
                    timestamp_front = adj_list['CAM_FRONT'][count]['timestamp']
                    pose_record = nuscenes.get(
                        'ego_pose', adj_list['CAM_FRONT'][count]['ego_pose_token'])

                    cam_infos = dict(
                        CAM_FRONT=dict(data_path=adj_list['CAM_FRONT'][count]['data_path']))
                    for cam in CAMS:
                        timestamp_curr_list = np.array(
                            [t['timestamp'] for t in adj_list[cam]], dtype=np.long)
                        diff = np.abs(timestamp_curr_list - timestamp_front)
                        selected_idx = np.argmin(diff)
                        cam_infos[cam] = dict(
                            data_path=adj_list[cam][int(selected_idx)]['data_path'])
                    sweeps.append(dict(
                        timestamp=timestamp_front,
                        cams=cam_infos,
                        ego2global_translation=pose_record['translation'],
                        ego2global_rotation=pose_record['rotation']))
                dataset['infos'][idx][adj] = sweeps if len(sweeps) > 0 else None

            # ego speed; move target velocity from global into ego-relative frame
            previous_id = idx
            if not sample['prev'] == '':
                previous_id = map_token_to_id[
                    nuscenes.get('sample', sample['prev'])['token']]
            next_id = idx
            if not sample['next'] == '':
                next_id = map_token_to_id[
                    nuscenes.get('sample', sample['next'])['token']]
            time_pre = 1e-6 * dataset['infos'][previous_id]['timestamp']
            time_next = 1e-6 * dataset['infos'][next_id]['timestamp']
            time_diff = time_next - time_pre
            posi_pre = np.array(
                dataset['infos'][previous_id]['ego2global_translation'], dtype=np.float32)
            posi_next = np.array(
                dataset['infos'][next_id]['ego2global_translation'], dtype=np.float32)
            velocity_global = (posi_next - posi_pre) / time_diff

            l2e_r_mat = Quaternion(info['lidar2ego_rotation']).rotation_matrix
            e2g_r_mat = Quaternion(info['ego2global_rotation']).rotation_matrix
            velocity_global = np.array([*velocity_global[:2], 0.0])
            velocity_lidar = velocity_global @ np.linalg.inv(e2g_r_mat).T \
                @ np.linalg.inv(l2e_r_mat).T
            velocity_lidar = velocity_lidar[:2]

            dataset['infos'][idx]['velo'] = velocity_lidar
            if split in ['train', 'val']:
                dataset['infos'][idx]['gt_velocity'] = \
                    dataset['infos'][idx]['gt_velocity'] - velocity_lidar.reshape(1, 2)

        filename = osp.join(
            root_path,
            'nuscenes_infos_%s_4d_interval%d_max%d.pkl' % (split, interval, max_adj))
        with open(filename, 'wb') as fid:
            pickle.dump(dataset, fid)
        print('=> wrote %s' % filename)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build temporal (4d) nuScenes info files')
    parser.add_argument('--root-path', type=str, default='./data/nuscenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini')
    parser.add_argument('--sets', type=str, nargs='+', default=['train', 'val'])
    parser.add_argument('--interval', type=int, default=3)
    parser.add_argument('--max-adj', type=int, default=60)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    add_adj_info(args.root_path, args.version, args.sets, args.interval, args.max_adj)
