from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re
import warnings

from .. import ImageDataset


class WILDTRACK(ImageDataset):
    """WILDTRACK.

    Reference:
        Chavdarova et al. WILDTRACK: A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection.

    URL: `<https://www.epfl.ch/labs/cvlab/data/data-wildtrack/>`_
    
    Dataset statistics:
        - identities: 313.
        - images: 3848  (train) + 695  (query) + 3153  (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'WILDTRACK'
    dataset_url = ''

    def __init__(self, root='', market1501_500k=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'WILDTRACK')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "train" under '
                          '"WILDTRACK".')
        
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(WILDTRACK, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        ####### modified #######
        pid2label = {pid:label for label, pid in enumerate(sorted(pid_container))}
        # pid2label = {pid:label for label, pid in enumerate(pid_container)}

        ########################

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1200  # pid == 0 means background
            assert 1 <= camid <= 7
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data