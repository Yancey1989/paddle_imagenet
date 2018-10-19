""" /*************************************************************************** * * Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
 * 
 **************************************************************************/
 
/**
 * @file imagenet_demo.py
 * @author wanglong(com@baidu.com)
 * @date 2018/05/24 16:12:09
 * @brief  a demo to show how to build readers with preprocessing pipeline on imagenet dataset
 **/
"""

import sys
import os
import time
import logging
#path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../')
#if path not in sys.path:
#    sys.path.insert(0, path)

import datareader
import datareader.data_source.source as source
import datareader.transformer.imagenet_transformer as transformer
datareader.set_loglevel(logging.DEBUG)
from PIL import Image
import io

#more detail info about these datasets can be found here:
# http://aiflow.baidu.com/#/aidatarepo/index?project_id=20&domain_id=8
def afs_record_parser(r):
    """ parse kv data from sequence file for imagenet
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    #maybe we need to normalize this data to be in one schema
    #the schema in label is:
    #   [<line_num>, <contsign>, <file_name>, <cate_id>, <cate_name>, <cate_desc>]
    if len(obj['label']) >= 4:
        label = int(obj['label'][3]) # class id
    else:
        label = int(obj['label'][2])

    image = obj['image'] #binary jpeg
    #image = Image.open(io.BytesIO(image)).resize((160, 160), resample=Image.BICUBIC)
    #out = io.BytesIO()
    #image.save(out, format='jpeg')
    #return out.getvalue(), label
    return image, label


afs_repo = 'afs://vis-temp,vis-temp123456@xingtian.afs.baidu.com:9902/user/vis-temp/datasets'
drepo_prefix = 'drepo:/token@'
drepo_conf = {
    'imagenet100': {
        'train': drepo_prefix + 'ds-f28d5353-9ac4-439b-b16b-8b6b4fe148a9',
        'val': drepo_prefix + 'ds-5afe71f2-8184-41b0-b207-deb2ddb288b4',
        'test': drepo_prefix + 'ds-5afe71f2-8184-41b0-b207-deb2ddb288b4',
        'parser': lambda r: (r['image'], int(r['label']))
    },
    'imagenet': {
        'train': drepo_prefix + 'ds-33113d7e-1786-409c-bed8-47f3e63070b8',
        'val': drepo_prefix + 'ds-81fea968-e131-49e0-a29b-4bf306c8c22c',
        'test': drepo_prefix + 'ds-933a4b21-2757-4e7a-8897-da9ca3bc0fdf',
        'parser': lambda r: (r['image'], int(r['label']))
    },
    'afs_imagenet100': {
        'train': afs_repo + '/imagenet100/train',
        'val': afs_repo + '/imagenet100/val',
        'test': afs_repo + '/imagenet100/test',
        'parser': afs_record_parser
    },
    'afs_imagenet': {
        'train': afs_repo + '/imagenet/train',
        'val': afs_repo + '/imagenet/val',
        'test': afs_repo + '/imagenet/test',
        'parser': afs_record_parser
    },
}

g_settings = {'name': 'imagenet', 'part_id': 0, \
                'part_num': 1, 'cache': None, \
                'crop': 128, 'concurrency': 16, \
                'resize': -1,
                'buffer_size': 10000, 'accelerate': True, \
                'shuffle_size': 100000, 'infinite': False}


def make_reader(mode, uri, parser, cache=None, \
        part_id=None, part_num=None, infinite=False, **kwargs):
    """ make a reader for data from uri
    """
    sc = source.load(uri=uri, part_id=part_id, part_num=part_num, 
            cache=cache, infinite=infinite)

    default_conf = {
            'crop': g_settings['crop'],
            'concurrency': g_settings['concurrency'],
            'buffer_size': g_settings['buffer_size'],
            'accelerate': g_settings['accelerate'],
            'shuffle_size': g_settings['shuffle_size'] if mode == 'train' else 0,
            'resize': g_settings['resize'],
            }

    for k, v in default_conf.items():
        if k not in kwargs:
            kwargs[k] = v

    p = transformer.build_pipeline(
            mode=mode,
            pre_maps=[parser],
            **kwargs)
    return p.transform(sc.reader())


def train(name=None, parser=None, part_id=None, part_num=None, 
        cache=None, infinite=None, **kwargs):
    """ make reader for train dataset
    """
    name = g_settings['name'] if name is None else name
    part_id = g_settings['part_id'] if part_id is None else part_id
    part_num = g_settings['part_num'] if part_num is None else part_num
    cache = g_settings['cache'] if cache is None else cache
    infinite = g_settings['infinite'] if infinite is None else infinite

    uri = drepo_conf[name]['train']
    parser = drepo_conf[name]['parser'] if parser is None else parser
    return make_reader('train', uri, parser, cache=cache,
            part_id=part_id, part_num=part_num, infinite=infinite, **kwargs)


def val(name=None, parser=None, cache=None, **kwargs):
    """ make reader for val dataset
    """
    name = g_settings['name'] if name is None else name
    cache = g_settings['cache'] if cache is None else cache

    uri = drepo_conf[name]['val']
    parser = drepo_conf[name]['parser'] if parser is None else parser
    return make_reader('val', uri, parser, cache=cache, **kwargs)


def test(name=None, parser=None, cache=None, **kwargs):
    """ make reader for test dataset
    """
    name = g_settings['name'] if name is None else name
    cache = g_settings['cache'] if cache is None else cache

    uri = drepo_conf[name]['test']
    parser = drepo_conf[name]['parser'] if parser is None else parser
    return make_reader('test', uri, parser, cache=cache, **kwargs)


def main():
    """ test readers for datasets
    """
    name = 'afs_imagenet'
    train_reader = train(name=name, part_id=0, part_num=4,
            cache='/data_cache')

    val_reader = val(name=name, cache='/data_cache')
    #test_reader = test(name=name, cache='/data_cache')

    ct = 0
    start_ts = time.time()
    #for img, label in val_reader():
    #    assert img.shape == (3, 224, 224)
    #    if ct % 1000 == 0:
    #        cost = 1000 * (time.time() - start_ts)
    #        start_ts = time.time()
    #        print('read %d samples in %dms' % (ct, cost))
    #    ct += 1
    #    #break
    #print('total got %d samples' % ct)

    ct = 0
    start_ts = time.time()
    #for img, label in test_reader():
    #    assert img.shape == (3, 224, 224)
    #    if ct % 1000 == 0:
    #        cost = 1000 * (time.time() - start_ts)
    #        start_ts = time.time()
    #        print('read %d samples in %dms' % (ct, cost))
    #    ct += 1
    #    #break

    print('total got %d samples' % ct)
    start_ts = time.time()
    #maybe need to wait several minutes for large range shuffle in this
    for img, label in train_reader():
        #print('debug', img.shape)
        #assert img.shape == (3, 224, 224)
        if ct % 1000 == 0:
            cost = 1000 * (time.time() - start_ts)
            start_ts = time.time()
            print('read %d samples in %dms' % (ct, cost))
        ct += 1
        #break

    print('total got %d samples' % ct)


if __name__ == "__main__":
    exit(main())

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
