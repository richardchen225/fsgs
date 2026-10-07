import os

from .dataset_dl3dv import DatasetDL3DV


class DatasetDL3DV1(DatasetDL3DV):
    """DL3DV layout with one directory between a scene and nerfstudio.

    Expected layout:
        root/<index item>/<middle directory>/nerfstudio/
            transforms.json
            images_8/
    """

    @staticmethod
    def resolve_scene_dir(data_root, item):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            for middle_name in sorted(os.listdir(item_path)):
                middle_path = os.path.join(item_path, middle_name)
                if not os.path.isdir(middle_path):
                    continue
                scene_path = os.path.join(middle_path, "nerfstudio")
                images_8_path = os.path.join(scene_path, "images_8")
                transforms_path = os.path.join(scene_path, "transforms.json")
                if os.path.isdir(images_8_path) and os.path.isfile(
                    transforms_path
                ):
                    return scene_path, images_8_path, transforms_path

        # Keep the original layouts valid as a fallback.
        return DatasetDL3DV.resolve_scene_dir(data_root, item)
