# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import cProfile
import time
import os
import traceback

import numpy as np

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler
import paddle.fluid.transpiler.distribute_transpiler as distribute_transpiler

from args import *
import torchvision_reader

DEBUG_PROG = bool(os.getenv("DEBUG_PROG", "0"))

def test_parallel(exe, test_args, args, test_prog, feeder, bs):
    acc_evaluators = []
    for i in xrange(len(test_args[2])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[2]]
    print(to_fetch)
    test_args[4].start()
    batch_id = 0
    start_ts = time.time()
    while True:
        try:
            acc_rets = exe.run(fetch_list=to_fetch)
            ret_result = [np.mean(np.array(ret)) for ret in acc_rets]
            print("Test batch: [%d], acc_rets: [%s]" % (batch_id, ret_result))
            batch_id += 1
            for i, e in enumerate(acc_evaluators):
                e.update(
                    value=np.array(acc_rets[i]), weight=bs)
        except fluid.core.EOFException as eof:
            test_args[4].reset()
            break
    num_samples = batch_id * bs * args.gpus 
    print_train_time(start_ts, time.time(), num_samples)

    return [e.eval() for e in acc_evaluators]

def refresh_program(args, epoch, sz, rsz, bs, need_update_start_prog=False, min_scale=0.08):
    print('program changed: epoch: [%d], image size: [%d], resize: [%d], batch_size:[%d]' % (epoch, sz, rsz, bs))
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    py_reader_startup_prog = fluid.Program()

    model_def = __import__("models.%s" % args.model, fromlist=["models"])

    train_args = list(model_def.get_model(args, True, train_prog, startup_prog, py_reader_startup_prog, sz, rsz, bs, min_scale))
    test_args = list(model_def.get_model(args, False, test_prog, startup_prog, py_reader_startup_prog, sz, rsz, bs, min_scale))

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    startup_exe = fluid.Executor(place)
    print("execute py_reader startup program")
    startup_exe.run(py_reader_startup_prog)

    if need_update_start_prog:
        print("execute startup program")
        startup_exe.run(startup_prog)
        if args.init_conv2d_kaiming:
            import torch
            conv2d_w_vars = [var for var in startup_prog.global_block().vars.values() if var.name.startswith('conv2d_')]
            for var in conv2d_w_vars:
                torch_w = torch.empty(var.shape)
                print("initialize %s, shape: %s, with kaiming normalization." % (var.name, var.shape))
                kaiming_np = torch.nn.init.kaiming_normal_(torch_w, mode='fan_out', nonlinearity='relu').numpy()
                tensor = fluid.global_scope().find_var(var.name).get_tensor()
                tensor.set(np.array(kaiming_np, dtype='float32'), place)
        
        if args.use_uint8_reader:
            img_mean_np = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255]).astype("float32").reshape((3, 1, 1))
            img_std_np = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255]).astype("float32").reshape((3, 1, 1))
            mean_var = fluid.global_scope().find_var("img_mean")
            mean_var.get_tensor().set(img_mean_np, place)
            std_var = fluid.global_scope().find_var("img_std")
            std_var.get_tensor().set(img_std_np, place)


    if DEBUG_PROG:
        with open('/tmp/train_prog_pass%d' % epoch, 'w') as f: f.write(train_prog.to_string(True))
        with open('/tmp/test_prog_pass%d' % epoch, 'w') as f: f.write(test_prog.to_string(True))
        with open('/tmp/startup_prog_pass%d' % epoch, 'w') as f: f.write(startup_prog.to_string(True))
        with open('/tmp/py_reader_startup_prog_pass%d' % epoch, 'w') as f: f.write(py_reader_startup_prog.to_string(True))

    strategy = fluid.ExecutionStrategy()
    strategy.num_threads = args.cpus
    strategy.allow_op_delay = False
    build_strategy = fluid.BuildStrategy()
    build_strategy.reduce_strategy = fluid.BuildStrategy().ReduceStrategy.Reduce \
        if args.reduce_strategy == "reduce" else fluid.BuildStrategy().ReduceStrategy.AllReduce

    avg_loss = train_args[0]
    train_exe = fluid.ParallelExecutor(
        True,
        avg_loss.name,
        main_program=train_prog,
        exec_strategy=strategy,
        build_strategy=build_strategy)

    test_exe = fluid.ParallelExecutor(
        True, main_program=test_prog, share_vars_from=train_exe)

    return train_args, test_args, test_prog, train_exe, test_exe



# NOTE: only need to benchmark using parallelexe
def train_parallel(args):
    over_all_start = time.time()
    test_prog = fluid.Program()

    trainer_id = 0
    exe = None
    test_exe = None
    bs = 224
    train_args = None
    test_args = None
    for pass_id in range(args.pass_num):
        # program changed
        if pass_id == 0:
            train_args, test_args, test_prog, exe, test_exe = refresh_program(args, pass_id, sz=128, rsz=160, bs=bs, need_update_start_prog=True)
        elif pass_id == 13:
            bs = 96
            train_args, test_args, test_prog, exe, test_exe = refresh_program(args, pass_id, sz=224, rsz=352, bs=bs, min_scale=0.087)
        elif pass_id == 25:
            bs = 50
            train_args, test_args, test_prog, exe, test_exe = refresh_program(args, pass_id, sz=288, rsz=352, bs=bs, min_scale=0.5)
        else:
            pass

        avg_loss = train_args[0]
        num_samples = 0
        iters = 0
        start_time = time.time()
        train_args[4].start() # start pyreader
        while True:
            if args.profile and iters == 99:
                profiler.start_profiler("All")
                profiler.reset_profiler()
            elif args.profile and iters == 200:
                print("profiling total time: ", time.time() - start_time)
                profiler.stop_profiler("total", "/tmp/profile_%d_pass%d" %
                                       (trainer_id, pass_id))
            if iters == args.iterations:
                if args.use_reader_op:
                    train_args[4].reset()
                break

            if iters == args.skip_batch_num:
                start_time = time.time()
                num_samples = 0
            fetch_list = [avg_loss.name]
            acc_name_list = [v.name for v in train_args[2]]
            fetch_list.extend(acc_name_list)
            fetch_list.append("learning_rate")
            if iters % args.log_period == 0:
                should_print = True
            else:
                should_print = False

            fetch_ret = []
            try:
                if should_print:
                    fetch_ret = exe.run(fetch_list)
                else:
                    exe.run([])
            except fluid.core.EOFException as eof:
                print("Finish current epoch, will reset pyreader...")
                train_args[4].reset()
                break
            except fluid.core.EnforceNotMet as ex:
                traceback.print_exc()
                exit(1)

            num_samples += bs * args.gpus

            if should_print:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print("Pass %d, batch %d, loss %s, accucacys: %s, learning_rate %s, py_reader queue_size: %d" %
                      (pass_id, iters, fetched_data[0], fetched_data[1:-1], fetched_data[-1], train_args[4].queue.size()))
            iters += 1

        print_train_time(start_time, time.time(), num_samples)

        test_ret = test_parallel(test_exe, test_args, args, test_prog,
                                 None, bs)
        print("Pass: %d, Test Accuracy: %s, Spend %.2f hours\n" %
            (pass_id, [np.mean(np.array(v)) for v in test_ret], (time.time() - over_all_start) / 3600))

    print("total train time: ", time.time() - over_all_start)


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('----------- Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def print_train_time(start_time, end_time, num_samples):
    train_elapsed = end_time - start_time
    examples_per_sec = num_samples / train_elapsed
    print('\nTotal examples: %d, total time: %.5f, %.5f examples/sed\n' %
          (num_samples, train_elapsed, examples_per_sec))


def print_paddle_envs():
    print('----------- Configuration envs -----------')
    for k in os.environ:
        if "PADDLE_" in k:
            print "ENV %s:%s" % (k, os.environ[k])
    print('------------------------------------------------')


def main():
    args = parse_args()
    print_arguments(args)
    print_paddle_envs()
    if args.no_random:
        fluid.default_startup_program().random_seed = 1

    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

    if args.device == "CPU":
        raise Exception("Only support GPU perf with parallel exe")
    train_parallel(args)


if __name__ == "__main__":
    main()
