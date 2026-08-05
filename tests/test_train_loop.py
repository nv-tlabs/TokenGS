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

import contextlib
import io
import unittest
from unittest import mock

import torch
from accelerate import Accelerator

from tokengs import train as train_module
from tokengs.options import Options


class FakeScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def get_last_lr(self):
        return [0.0]


class FakeDataset:
    def __init__(self):
        self.rng_epochs = []

    def set_rng_epoch(self, epoch):
        self.rng_epochs.append(epoch)


class DotProductModel(torch.nn.Module):
    """Loss is `weight . batch`, so a micro-batch gradient is exactly its own input."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))

    def forward(self, data):
        loss = (self.weight * data).sum()
        return {"loss": loss, "psnr": torch.zeros(()), "loss_rgb": loss.detach()}


class RecordingSGD(torch.optim.SGD):
    """Records the gradient each real (non-skipped) `step()` is about to consume."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.seen_grads = []

    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                self.seen_grads.append(None if param.grad is None else param.grad.detach().clone())
        return super().step(closure)


def run_train_epochs(epoch_batches, gradient_accumulation_steps=2):
    """Drive the real `train_epoch` on CPU and return the gradient seen at each update.

    `epoch_batches` is one list of scalar batch values per epoch.
    """
    torch.manual_seed(0)
    model = DotProductModel()
    # lr=0 keeps the weights fixed; the gradient does not depend on them either way.
    optimizer = RecordingSGD(model.parameters(), lr=0.0)
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, cpu=True)
    prepared_model, prepared_optimizer = accelerator.prepare(model, optimizer)

    opt = Options(
        gradient_accumulation_steps=gradient_accumulation_steps,
        # Clipping would rescale the very quantity under test.
        gradient_clip=1e9,
        use_wandb=False,
    )
    dataset = FakeDataset()

    # train_epoch reports CUDA memory and dumps images on iteration 0 regardless of
    # the configured frequencies; neither is part of what these tests assert.
    with mock.patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 1 << 30)), \
            mock.patch.object(train_module, "log_training_images"), \
            contextlib.redirect_stdout(io.StringIO()):
        for epoch, values in enumerate(epoch_batches):
            dataloader = [torch.full((4,), value) for value in values]
            train_module.train_epoch(
                opt,
                accelerator,
                prepared_model,
                prepared_optimizer,
                FakeScheduler(),
                dataloader,
                len(dataloader),
                epoch,
                None,
                0.0,
                dataset,
            )

    return [None if grad is None else grad.flatten()[0].item() for grad in optimizer.seen_grads]


class TestGradientAccumulation(unittest.TestCase):
    def test_update_sees_every_micro_batch_in_the_cycle(self):
        # accelerate scales each micro-batch loss by 1/accumulation -> (1 + 2)/2.
        # Zeroing at the top of the step instead would leave only the last one, 2/2.
        seen = run_train_epochs([[1.0, 2.0]])
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], 1.5, places=5)

    def test_accumulation_of_one_is_unaffected(self):
        seen = run_train_epochs([[1.0, 2.0]], gradient_accumulation_steps=1)
        self.assertEqual(len(seen), 2)
        self.assertAlmostEqual(seen[0], 1.0, places=5)
        self.assertAlmostEqual(seen[1], 2.0, places=5)

    def test_partial_cycle_does_not_leak_into_the_next_epoch(self):
        # Epoch 0 has an odd number of batches, so it ends mid-cycle holding 1.0/2.
        # Epoch 1's first update must see only its own micro-batch, 10.0/2.
        seen = run_train_epochs([[1.0, 1.0, 1.0], [10.0, 10.0]])
        self.assertEqual(len(seen), 2)
        self.assertAlmostEqual(seen[0], 1.0, places=5)
        self.assertAlmostEqual(seen[1], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
