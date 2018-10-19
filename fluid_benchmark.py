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

DEBUG_PROG = bool(os.getenv("DEBUG_PROG", "0"))

def append_nccl2_prepare(trainer_id, startup_prog):
    if trainer_id >= 0:
        # append gen_nccl_id at the end of startup program
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
        port = os.getenv("PADDLE_PSERVER_PORT")
        worker_ips = os.getenv("PADDLE_TRAINER_IPS")
        worker_endpoints = []
        for ip in worker_ips.split(","):
            worker_endpoints.append(':'.join([ip, port]))
        num_trainers = len(worker_endpoints)
        current_endpoint = os.getenv("PADDLE_CURRENT_IP") + ":" + port
        worker_endpoints.remove(current_endpoint)

        nccl_id_var = startup_prog.global_block().create_var(
            name="NCCLID",
            persistable=True,
            type=fluid.core.VarDesc.VarType.RAW)
        startup_prog.global_block().append_op(
            type="gen_nccl_id",
            inputs={},
            outputs={"NCCLID": nccl_id_var},
            attrs={
                "endpoint": current_endpoint,
                "endpoint_list": worker_endpoints,
                "trainer_id": trainer_id
            })
        return nccl_id_var, num_trainers, trainer_id
    else:
        raise Exception("must set positive PADDLE_TRAINER_ID env variables for "
                        "nccl-based dist train.")


def dist_transpile(trainer_id, args, train_prog, startup_prog):
    if trainer_id < 0:
        return None, None

    # the port of all pservers, needed by both trainer and pserver
    port = os.getenv("PADDLE_PSERVER_PORT", "6174")
    # comma separated ips of all pservers, needed by trainer and
    # pserver
    pserver_ips = os.getenv("PADDLE_PSERVER_IPS", "")
    eplist = []
    for ip in pserver_ips.split(","):
        eplist.append(':'.join([ip, port]))
    pserver_endpoints = ",".join(eplist)
    # total number of workers/trainers in the job, needed by
    # trainer and pserver
    trainers = int(os.getenv("PADDLE_TRAINERS"))
    # the IP of the local machine, needed by pserver only
    current_endpoint = os.getenv("PADDLE_CURRENT_IP", "") + ":" + port
    # the role, should be either PSERVER or TRAINER
    training_role = os.getenv("PADDLE_TRAINING_ROLE")

    config = distribute_transpiler.DistributeTranspilerConfig()
    config.slice_var_up = not args.no_split_var
    t = distribute_transpiler.DistributeTranspiler(config=config)
    t.transpile(
        trainer_id,
        # NOTE: *MUST* use train_prog, for we are using with guard to
        # generate different program for train and test.
        program=train_prog,
        pservers=pserver_endpoints,
        trainers=trainers,
        sync_mode=not args.async_mode,
        startup_program=startup_prog)
    if training_role == "PSERVER":
        pserver_program = t.get_pserver_program(current_endpoint)
        pserver_startup_program = t.get_startup_program(
            current_endpoint, pserver_program, startup_program=startup_prog)
        return pserver_program, pserver_startup_program
    elif training_role == "TRAINER":
        train_program = t.get_trainer_program()
        return train_program, startup_prog
    else:
        raise ValueError(
            'PADDLE_TRAINING_ROLE environment variable must be either TRAINER or PSERVER'
        )


def test_parallel(exe, test_args, args, test_prog, feeder):
    acc_evaluators = []
    for i in xrange(len(test_args[2])):
        acc_evaluators.append(fluid.metrics.Accuracy())

    to_fetch = [v.name for v in test_args[2]]
    if args.use_reader_op:
        test_args[4].start()
        while True:
            try:
                acc_rets = exe.run(fetch_list=to_fetch)
                for i, e in enumerate(acc_evaluators):
                    e.update(
                        value=np.array(acc_rets[i]), weight=args.batch_size)
            except fluid.core.EOFException as eof:
                test_args[4].reset()
                break
    else:
        for batch_id, data in enumerate(test_args[3]()):
            acc_rets = exe.run(feed=feeder.feed(data), fetch_list=to_fetch)
            for i, e in enumerate(acc_evaluators):
                e.update(value=np.array(acc_rets[i]), weight=len(data))

    return [e.eval() for e in acc_evaluators]

def refresh_program(args, epoch, sz, rsz, bs, need_update_start_prog=False):
    print('program changed: epoch: [%d], image size: [%d], resize: [%d], batch_size:[%d]' % (epoch, sz, rsz, bs))
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()
    py_reader_startup_prog = fluid.Program()

    model_def = __import__("models.%s" % args.model, fromlist=["models"])

    train_args = list(model_def.get_model(args, True, train_prog, startup_prog, py_reader_startup_prog, sz, rsz, bs))
    test_args = list(model_def.get_model(args, False, test_prog, startup_prog, py_reader_startup_prog, sz, rsz, bs))

    place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    startup_exe = fluid.Executor(place)
    print("execute py_reader startup program")
    startup_exe.run(py_reader_startup_prog)

    if need_update_start_prog:
        print("execute startup program")
        startup_exe.run(startup_prog)

    if DEBUG_PROG:
        with open('/tmp/train_prog_pass%d' % epoch, 'w') as f: f.write(train_prog.to_string(True))
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
def train_parallel(train_args, test_args, args, train_prog, test_prog,
                   startup_prog, nccl_id_var, num_trainers, trainer_id):
    over_all_start = time.time()
    feeder = None
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    if nccl_id_var and trainer_id == 0:
        #FIXME(wuyi): wait other trainer to start listening
        time.sleep(30)
    
    if args.update_method == "pserver":
        # parameter server mode distributed training, merge
        # gradients on local server, do not initialize
        # ParallelExecutor with multi server all-reduce mode.
        num_trainers = 1
        trainer_id = 0
    exe = None
    test_exe = None

    for pass_id in range(args.pass_num):
        # program changed
        if pass_id == 0:
            train_args, test_args, test_prog, exe, test_exe = refresh_program(args, pass_id, sz=128, rsz=160, bs=224, need_update_start_prog=True)
        elif pass_id == 13:
            train_args, test_args, test_prog, exe, test_exe = refresh_program(args, pass_id, sz=224, rsz=352, bs=96)
        else:
            pass

        avg_loss = train_args[0]
        num_samples = 0
        iters = 0
        start_time = time.time()
        if not args.use_reader_op:
            reader_generator = train_args[3]()  #train_reader
        batch_id = 0
        data = None
        if args.use_reader_op:
            train_args[4].start()
        while True:
            if not args.use_reader_op and not args.use_fake_data:
                data = next(reader_generator, None)
                if data == None:
                    break
            if args.profile and batch_id == 99:
                profiler.start_profiler("All")
                profiler.reset_profiler()
            elif args.profile and batch_id == 200:
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

            if args.use_fake_data or args.use_reader_op:
                try:
                    fetch_ret = exe.run(fetch_list)
                except fluid.core.EOFException as eof:
                    break
                except fluid.core.EnforceNotMet as ex:
                    traceback.print_exc()
                    break
            else:
                fetch_ret = exe.run(fetch_list, feed=feeder.feed(data))
            if args.use_reader_op or args.use_fake_data:
                num_samples += args.batch_size * args.gpus
            else:
                num_samples += len(data)

            iters += 1
            if batch_id % 1 == 0:
                fetched_data = [np.mean(np.array(d)) for d in fetch_ret]
                print("Pass %d, batch %d, loss %s, accucacys: %s" %
                      (pass_id, batch_id, fetched_data[0], fetched_data[1:]))
                if args.use_reader_op:
                    print("pyreader queue: ", train_args[4].queue.size())
            batch_id += 1

        print_train_time(start_time, time.time(), num_samples)
        if args.use_reader_op:
            train_args[4].reset()  # reset reader handle
        else:
            del reader_generator

        test_ret = test_parallel(test_exe, test_args, args, test_prog,
                                 None)
        print("Pass: %d, Test Accuracy: %s\n" %
            (pass_id, [np.mean(np.array(v)) for v in test_ret]))

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

    # the unique trainer id, starting from 0, needed by trainer
    # only
    nccl_id_var, num_trainers, trainer_id = (
        None, 1, int(os.getenv("PADDLE_TRAINER_ID", "0")))

    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

    model_def = __import__("models.%s" % args.model, fromlist=["models"])

    train_prog = fluid.Program()
    test_prog = fluid.Program()
    startup_prog = fluid.Program()

    #train_args = list(model_def.get_model(args, True, train_prog, startup_prog, 0))
    #test_args = list(model_def.get_model(args, False, test_prog, startup_prog, 0))
    train_args = []
    test_args = []


    all_args = [train_args, test_args, args]

    if args.update_method == "pserver":
        train_prog, startup_prog = dist_transpile(trainer_id, args, train_prog,
                                                  startup_prog)
        if not train_prog:
            raise Exception(
                "Must configure correct environments to run dist train.")
        all_args.extend([train_prog, test_prog, startup_prog])
        if args.gpus > 1 and os.getenv("PADDLE_TRAINING_ROLE") == "TRAINER":
            all_args.extend([nccl_id_var, num_trainers, trainer_id])
            train_parallel(*all_args)
        elif os.getenv("PADDLE_TRAINING_ROLE") == "PSERVER":
            # start pserver with Executor
            server_exe = fluid.Executor(fluid.CPUPlace())
            server_exe.run(startup_prog)
            server_exe.run(train_prog)
        exit(0)

    # for other update methods, use default programs
    all_args.extend([train_prog, test_prog, startup_prog])

    if args.update_method == "nccl2":
        nccl_id_var, num_trainers, trainer_id = append_nccl2_prepare(
            trainer_id, startup_prog)

    if args.device == "CPU":
        raise Exception("Only support GPU perf with parallel exe")
    all_args.extend([nccl_id_var, num_trainers, trainer_id])
    train_parallel(*all_args)


if __name__ == "__main__":
    main()
