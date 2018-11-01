#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import time
import os
import math

import cProfile, pstats, StringIO

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
#import reader_fast
import datareader

## visreader for imagenet
import torchvision_reader

BN_NO_DECAY = bool(os.getenv("BN_NO_DECAY", "1"))

class ResNet():
    def __init__(self, layers=50, is_train=True):
        self.layers = layers
        self.is_train = is_train

    def net(self, input, class_dim=1000, img_size=224):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input, num_filters=64, filter_size=7, stride=2, act='relu')
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=3,
            pool_stride=2,
            pool_padding=1,
            pool_type='max')

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1)
        pool_size = int(img_size / 32)
        pool = fluid.layers.pool2d(
            input=conv, pool_size=pool_size, pool_type='avg', global_pooling=True)
        out = fluid.layers.fc(input=pool,
                              size=class_dim,
                              act='softmax',
                              param_attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.NormalInitializer(0.0, 0.01),
                                  regularizer=fluid.regularizer.L2Decay(1e-4)),
                              bias_attr=fluid.ParamAttr(
                                  regularizer=fluid.regularizer.L2Decay(1e-4)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      bn_init_value=1.0):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            bias_attr=False,
            param_attr=fluid.ParamAttr(regularizer=fluid.regularizer.L2Decay(1e-4)))
        return fluid.layers.batch_norm(input=conv, act=act, is_test=not self.is_train,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Constant(bn_init_value),
                regularizer=None))

    def shortcut(self, input, ch_out, stride):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(input, ch_out, 1, stride)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride):
        conv0 = self.conv_bn_layer(
            input=input, num_filters=num_filters, filter_size=1, act='relu')
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu')
        # init bn-weight0
        conv2 = self.conv_bn_layer(
            input=conv1, num_filters=num_filters * 4, filter_size=1, act=None, bn_init_value=0.0)

        short = self.shortcut(input, num_filters * 4, stride)

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')


def _model_reader_dshape_classdim(args, is_train, sz=224, rsz=352, min_scale=0.08):
    reader = None
    if args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, sz, sz]
        else:
            dshape = [sz, sz, 3]
        if is_train:
            reader = torchvision_reader.train(
                traindir="/data/imagenet/sz/%d/train" % rsz, sz=sz, min_scale=min_scale, use_uint8_reader=args.use_uint8_reader)
        else:
            reader = torchvision_reader.test(
                valdir="/data/imagenet/sz/%d/validation" % rsz, bs=None, sz=sz, rect_val=False, use_uint8_reader=args.use_uint8_reader)
    else:
        raise ValueError("only support imagenet dataset.")

    return None, reader, dshape, class_dim

def lr_decay(lrs, epochs, bs, total_image):
    boundaries = []
    values = []
    import math
    for idx, epoch in enumerate(epochs):
        step = math.ceil(total_image * 1.0 / (bs[idx] * 8))
        ratio = (lrs[idx][1] - lrs[idx][0]) / (epoch[1] - epoch[0])
        lr_base = lrs[idx][0]
        for s in xrange(epoch[0], epoch[1]):
            if boundaries:
                boundaries.append(boundaries[-1] + step)
            else:
                boundaries = [step]
            values.append(lr_base + ratio * (s - epoch[0]))
    values.append(lrs[-1])
    return boundaries, values

def get_model(args, is_train, main_prog, startup_prog, py_reader_startup_prog, sz, rsz, bs, min_scale):

    _, reader, dshape, class_dim = _model_reader_dshape_classdim(args,
                                                                 is_train,
                                                                 sz=sz,
                                                                 rsz=rsz,
                                                                 min_scale=min_scale)

    pyreader = None
    trainer_count = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            with fluid.program_guard(main_prog, py_reader_startup_prog):
                with fluid.unique_name.guard():
                    if args.use_uint8_reader:
                        img_dtype = 'uint8'
                    else:
                        img_dtype = 'float32'
                    pyreader = fluid.layers.py_reader(
                        capacity=bs * args.gpus,
                        shapes=([-1] + dshape, (-1, 1)),
                        dtypes=(img_dtype, 'int64'),
                        name="train_reader_" + str(sz) if is_train else "test_reader_" + str(sz),
                        use_double_buffer=True)
            input, label = fluid.layers.read_file(pyreader)
            if args.use_uint8_reader:
                cast = fluid.layers.cast(input, "float32")
                img_mean = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_mean", persistable=True)
                img_std = fluid.layers.create_global_var([3, 1, 1], 0.0, "float32", name="img_std", persistable=True)
                # elementwise_op would broadcast the parameter
                # t1 = cast - img_mean.broadcast(batch_size, 3, image_size, image_size)
                # t2 = t1 / img_std.broadcast(batch_size, 3, image_size, image_size)
                t1 = fluid.layers.elementwise_sub(cast, img_mean, axis=1)
                t2 = fluid.layers.elementwise_div(t1, img_std, axis=1)
            else:
                t2 = input

            model = ResNet(is_train=is_train)
            predict = model.net(t2, class_dim=class_dim, img_size=sz)
            cost = fluid.layers.cross_entropy(input=predict, label=label)
            avg_cost = fluid.layers.mean(x=cost)

            batch_acc1 = fluid.layers.accuracy(input=predict, label=label, k=1)
            batch_acc5 = fluid.layers.accuracy(input=predict, label=label, k=5)

            # configure optimize
            optimizer = None
            if is_train:
                total_images = 1281167 / trainer_count

                epochs = [(0,7), (7,13), (13, 22), (22, 25), (25, 28)]
                bs_epoch = [224, 224, 96, 96, 50]
                lrs = [(1.0, 2.0), (2.0, 0.25), (0.42857, 0.042857), (0.042857, 0.0042857), (0.00223, 0.000223), 0.000223]
                boundaries, values = lr_decay(lrs, epochs, bs_epoch, total_images)

                print("lr linear decay boundaries: ", boundaries, " \nvalues: ", values)
                optimizer = fluid.optimizer.Momentum(
                    learning_rate=fluid.layers.piecewise_decay(boundaries=boundaries, values=values),
                    momentum=0.9,
                    regularization=fluid.regularizer.L2Decay(1e-4))
                optimizer.minimize(avg_cost)
                training_role = os.getenv("PADDLE_TRAINING_ROLE")
                if args.memory_optimize and (training_role == "TRAINER" or args.update_method == "local"):
                    fluid.memory_optimize(main_prog, skip_grads=True)

    # config readers
    batched_reader = None
    pyreader.decorate_paddle_reader(paddle.batch(reader, batch_size=bs))

    return avg_cost, optimizer, [batch_acc1,
                                 batch_acc5], batched_reader, pyreader, py_reader_startup_prog