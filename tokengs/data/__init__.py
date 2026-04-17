# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
from tokengs.data.provider import Provider
from tokengs.data.sampler import FixedRandomSampler

class RNGConcatDataset(torch.utils.data.ConcatDataset):
    def set_rng_epoch(self, epoch):
        for dataset in self.datasets:
            if hasattr(dataset, 'set_rng_epoch'):
                dataset.set_rng_epoch(epoch)

def get_multi_dataloader(opt, accelerator):
    train_datasets, test_datasets = get_datasets(opt, accelerator)
    train_dataset = RNGConcatDataset(train_datasets)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(opt.seed),
    )
    
    test_dataset = RNGConcatDataset(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=FixedRandomSampler(test_dataset, seed=42) if not opt.evaluating else None,
    )

    # Return both dataloaders and the underlying datasets
    return train_dataloader, test_dataloader, train_dataset, test_dataset

def get_datasets(opt, accelerator):
    train_datasets = []
    test_datasets = []

    for idx in range(len(opt.data_mode)):
        begin_time = time.time()
        if isinstance(opt.data_mode[idx], str):
            dataset_name, num_repeat = opt.data_mode[idx], 1
        else:
            dataset_name, num_repeat = opt.data_mode[idx]

        train_dataset = Provider(dataset_name, opt, training=True, num_repeat=num_repeat)
        train_datasets.append(train_dataset)

        test_dataset = Provider(dataset_name, opt, training=False, num_repeat=num_repeat)
        test_datasets.append(test_dataset)
        if accelerator.is_main_process:
            print(f"Loaded {dataset_name}, train size: {len(train_dataset)}, test size: {len(test_dataset)}, loading took {time.time() - begin_time} seconds")

    return train_datasets, test_datasets
