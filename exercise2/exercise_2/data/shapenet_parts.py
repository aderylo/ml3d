from pathlib import Path
import json

import numpy as np
import torch


class ShapeNetParts(torch.utils.data.Dataset):
    num_classes = 50  # We have 50 parts classes to segment
    num_points = 1024
    dataset_path = Path("exercise_2/data/shapenetcore_partanno_segmentation_benchmark_v0/")  # path to point cloud data
    class_name_mapping = json.loads(Path("exercise_2/data/shape_parts_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())
    part_id_to_overall_id = json.loads(Path.read_text(Path(__file__).parent.parent / 'data' / 'partid_to_overallid.json'))

    def __init__(self, split):
        assert split in ['train', 'val', 'overfit']

        self.items = Path(f"exercise_2/data/splits/shapenet_parts/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        item = self.items[index]

        pointcloud, segmentation_labels = ShapeNetParts.get_point_cloud_with_labels(item)

        return {
            'points': pointcloud,
            'segmentation_labels': segmentation_labels
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['segmentation_labels'] = batch['segmentation_labels'].to(device)

    @staticmethod
    def get_point_cloud_with_labels(shapenet_id):
        category_id, shape_id = shapenet_id.split('/')

        # Paths to point cloud and segmentation files
        points_file = ShapeNetParts.dataset_path / category_id / 'points' / f'{shape_id}.npy'
        seg_file = ShapeNetParts.dataset_path / category_id  / 'points_label' / f'{shape_id}.seg'

        # Load point cloud data
        pointcloud = np.load(points_file)
        segmentation_labels = np.loadtxt(seg_file, dtype=int)

        # Subsample to 1024 points
        num_points = pointcloud.shape[0]
        if num_points >= ShapeNetParts.num_points:
            idx = np.random.choice(num_points, ShapeNetParts.num_points, replace=False)
        else:
            idx = np.random.choice(num_points, ShapeNetParts.num_points, replace=True)
        pointcloud = pointcloud[idx, :]
        segmentation_labels = segmentation_labels[idx]

        # Convert local labels to global labels
        keys = [f"{category_id}_{label}" for label in segmentation_labels]
        global_labels = [ShapeNetParts.part_id_to_overall_id[key] for key in keys]
        global_labels = np.array(global_labels)

        # Transpose point cloud to shape [3, 1024]
        pointcloud = pointcloud.T

        return pointcloud, global_labels
