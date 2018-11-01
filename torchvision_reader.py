import os

import numpy as np
import math
import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.sampler import Sampler
import torchvision
import pickle
from tqdm import tqdm
import time
import multiprocessing

TRAINER_NUMS = int(os.getenv("PADDLE_TRAINER_NUM", "1"))
TRAINER_ID = int(os.getenv("PADDLE_TRAINER_ID", "0"))
epoch = 0

def paddle_data_loader(torch_dataset, concurrent=24, queue_size=3072, use_uint8_reader=False):
    data_queue = multiprocessing.Queue(queue_size)
    FINISH_EVENT = "FINISH_EVENT"

    def _worker_loop(dataset, indices, worker_id):
        cnt = 0
        for idx in indices:
            cnt += 1
            img, label = torch_dataset[idx]
            if use_uint8_reader:
                img = np.array(img).astype('uint8').transpose((2, 0, 1))
            else:
                img = np.array(img).astype('float32').transpose((2, 0, 1))
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                scale = 1.0 / 255.0
                img_mean = np.array(mean).reshape((3, 1, 1))
                img_std = np.array(std).reshape((3, 1, 1))
                img = (img * scale - img_mean) / img_std
            data_queue.put((img, label))
        print("worker: [%d] read [%d] samples. " % (worker_id, cnt))
        data_queue.put(FINISH_EVENT)

    def _reader():
        worker_processes = []
        total_img = len(torch_dataset)
        print("total image: ", total_img)
        indices = [i for i in xrange(total_img)]
        import random
        import math
        random.seed(time.time())
        random.shuffle(indices)
        print("shuffle indices: %s ..." % indices[:10])

        imgs_per_worker = int(math.ceil(total_img / concurrent))
        for i in xrange(concurrent):
            start = i * imgs_per_worker
            end = (i + 1) * imgs_per_worker if i != concurrent - 1 else None
            sliced_indices = indices[start:end]
            w = multiprocessing.Process(
                target=_worker_loop,
                args=(torch_dataset, sliced_indices, i)
            )
            w.daemon = True
            w.start()
            worker_processes.append(w)
        finish_workers = 0
        worker_cnt = len(worker_processes)
        while finish_workers < worker_cnt:
            sample = data_queue.get()
            if sample == FINISH_EVENT:
                finish_workers += 1
            else:
                yield sample

    return _reader

def train(traindir, sz, min_scale=0.08, use_uint8_reader=False):
    train_tfms = [
        transforms.RandomResizedCrop(sz, scale=(min_scale, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(train_tfms))
    return paddle_data_loader(train_dataset, queue_size=1024, use_uint8_reader=use_uint8_reader)

def test(valdir, bs, sz, rect_val=False, use_uint8_reader=False):
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, bs)

        ar_tfms = [transforms.Resize(int(sz* 1.14)), CropArTfm(idx2ar, sz)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        return paddle_data_loader(val_dataset, use_uint8_reader=use_uint8_reader)

    val_tfms = [transforms.Resize(int(sz* 1.14)), transforms.CenterCrop(sz)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    return paddle_data_loader(val_dataset, use_uint8_reader=use_uint8_reader)



def create_validation_set(valdir, batch_size, target_size, rect_val, distributed):
    if rect_val:
        idx_ar_sorted = sort_ar(valdir)
        idx_sorted, _ = zip(*idx_ar_sorted)
        idx2ar = map_idx2ar(idx_ar_sorted, batch_size)

        ar_tfms = [transforms.Resize(int(target_size * 1.14)), CropArTfm(idx2ar, target_size)]
        val_dataset = ValDataset(valdir, transform=ar_tfms)
        val_sampler = DistValSampler(idx_sorted, batch_size=batch_size, distributed=distributed)
        return val_dataset, val_sampler

    val_tfms = [transforms.Resize(int(target_size * 1.14)), transforms.CenterCrop(target_size)]
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose(val_tfms))
    val_sampler = DistValSampler(list(range(len(val_dataset))), batch_size=batch_size, distributed=distributed)
    return val_dataset, val_sampler


class ValDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(ValDataset, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            for tfm in self.transform:
                if isinstance(tfm, CropArTfm):
                    sample = tfm(sample, index)
                else:
                    sample = tfm(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DistValSampler(Sampler):
    # DistValSampler distrbutes batches equally (based on batch size) to every gpu (even if there aren't enough images)
    # WARNING: Some baches will contain an empty array to signify there aren't enough images
    # Distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = TRAINER_NUMS
            self.global_rank = TRAINER_NUMS
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per sample. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_batches = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total images / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_batches * self.batch_size

    def __iter__(self):
        offset = self.num_samples * self.global_rank
        sampled_indices = self.indices[offset:offset + self.num_samples]
        for i in range(self.expected_num_batches):
            offset = i * self.batch_size
            yield sampled_indices[offset:offset + self.batch_size]

    def __len__(self):
        return self.expected_num_batches

    def set_epoch(self, epoch):
        return


class CropArTfm(object):
    def __init__(self, idx2ar, target_size):
        self.idx2ar, self.target_size = idx2ar, target_size

    def __call__(self, img, idx):
        target_ar = self.idx2ar[idx]
        if target_ar < 1:
            w = int(self.target_size / target_ar)
            size = (w // 8 * 8, self.target_size)
        else:
            h = int(self.target_size * target_ar)
            size = (self.target_size, h // 8 * 8)
        return torchvision.transforms.functional.center_crop(img, size)


def sort_ar(valdir):
    idx2ar_file = valdir + '/../sorted_idxar.p'
    if os.path.isfile(idx2ar_file):
        return pickle.load(open(idx2ar_file, 'rb'))
    print('Creating AR indexes. Please be patient this may take a couple minutes...')
    val_dataset = datasets.ImageFolder(valdir)  # AS: TODO: use Image.open instead of looping through dataset
    sizes = [img[0].size for img in tqdm(val_dataset, total=len(val_dataset))]
    idx_ar = [(i, round(s[0] / s[1], 5)) for i, s in enumerate(sizes)]
    sorted_idxar = sorted(idx_ar, key=lambda x: x[1])
    pickle.dump(sorted_idxar, open(idx2ar_file, 'wb'))
    print('Done')
    return sorted_idxar

def chunks(l, n):
    n = max(1, n)
    return (l[i:i + n] for i in range(0, len(l), n))


def map_idx2ar(idx_ar_sorted, batch_size):
    ar_chunks = list(chunks(idx_ar_sorted, batch_size))
    idx2ar = {}
    for chunk in ar_chunks:
        idxs, ars = list(zip(*chunk))
        mean = round(np.mean(ars), 5)
        for idx in idxs:
            idx2ar[idx] = mean
    return idx2ar

if __name__ == "__main__":
    train_tfms = [
        transforms.RandomResizedCrop(128, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip()
    ]
    train_dataset = datasets.ImageFolder("/data/imagenet/sz/160/train", transforms.Compose(train_tfms))
    train_reader = paddle_data_loader(train_dataset)


    import time
    start_ts = time.time()
    for idx, data in enumerate(train_reader()):
        if (idx + 1) % 1000 == 0:
            cost = (time.time() - start_ts)
            print("%d samples per second" % (1000 / cost))
            start_ts = time.time()