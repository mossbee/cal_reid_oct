import os.path as osp
from .bases import BaseImageDataset


class ListReID(BaseImageDataset):
    """
    Dataset that reads training data from a list file with lines:
        <relative_image_path> <pid>
    All images are assumed to be from the same camera; camid is set to 0.
    PIDs are remapped to contiguous labels for training.
    """

    def __init__(self, root: str, list_path: str, verbose: bool = True, **kwargs):
        super(ListReID, self).__init__()
        self.root = root
        self.list_path = list_path

        train = self._process_train_list(self.root, self.list_path)
        query = []
        gallery = []

        if verbose:
            print("=> ListReID loaded from {}".format(self.list_path))
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = 0, 0, 0
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = 0, 0, 0

    def _process_train_list(self, image_root: str, list_path: str):
        entries = []
        with open(list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, pid_str = line.split()
                pid = int(pid_str)
                abs_path = osp.join(image_root, rel_path)
                # camid fixed to 0 (single camera)
                entries.append((abs_path, pid, 0))

        # relabel PIDs to contiguous range for training
        pid_set = sorted({pid for _, pid, _ in entries})
        pid2label = {pid: idx for idx, pid in enumerate(pid_set)}
        relabeled = [(p, pid2label[pid], cam) for (p, pid, cam) in entries]
        return relabeled


