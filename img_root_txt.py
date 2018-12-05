import os
import json
import random
import codecs
import numpy as np
import os.path as osp

root = '/home/shuhan/multi-source_ReID/fc_learn/datasets'
market_root = "/home/share/hongxing/Market"
duke_root = "/home/shuhan/ReID_data/DUKE/DukeMTMC-reID"

did_map = {'cuhk01':0, 'cuhk03':1, 'viper':2, 'ilids':3, '3dpes':4, 'market':5, 'duke':6}

def make_target_txt(target):
    # write each sample with: File_name pid cid did
    did = did_map[target]
    meta_f = open(os.path.join(root, target + '/meta.json'))
    meta = json.load(meta_f)
    identities = meta['identities']
    split_f = open(os.path.join(root, target + '/split.json'))
    split = json.load(split_f)

    train_id_map = {idx:i for i, idx in enumerate(split['trainval'])}
    train_list = []
    for id in split['trainval']:
        for views in identities[id]:
            train_list.append(views[:])
    train_out = get_list(train_list, train_id_map, target, did)
    write_list(train_out, os.path.join(root, target + '_train.txt'))

    test_id_map = {idx:i for i, idx in enumerate(split['test_probe'] + split['test_gallery'])}
    assert len(set(split['test_probe']) - set(split['test_gallery'])) == 0
    prob_list, gallery_list = [], []
    for id in split['test_probe']:
        views = identities[id]
        for idx, view in enumerate(views):
            if idx < len(views)//2:
                prob_list.append(view)
            else:
                gallery_list.append(view)
    only_in_gallery = list(set(split['test_gallery']) - set(split['test_probe']))

    for id in only_in_gallery:
        for views in identities[id]:
            gallery_list.append(views[:])

    prob_out = get_list(prob_list, test_id_map, target, did)
    gallery_out = get_list(gallery_list, test_id_map, target, did)
    write_list(prob_out, os.path.join(root, target + '_prob.txt'))
    write_list(gallery_out, os.path.join(root, target + '_gallery.txt'))


def make_market_txt(img_root, name):
    # write each sample with: File_name pid cid did for Market or Duke
    did = did_map[name]
    files = {}
    id_maps = {}
    for _, _, file in os.walk(os.path.join(img_root, 'train')):
        files['train'] = file
    for _, _, file in os.walk(os.path.join(img_root, 'test')):
        files['test'] = file
    for _, _, file in os.walk(os.path.join(img_root, 'query')):
        files['query'] = file
    for file in files['train']:
        if ".jpg" not in file:
            files['train'].remove(file)
    train_unique_id = set([int(file[:4]) for file in files['train']])
    test_ids = []
    for file in files['test']+files['query']:
        if file[:2] == '-1':
            test_ids.append(-1)
        else:
            test_ids.append(int(file[:4]))
    test_unique_id = set(test_ids)
    id_maps['train'] = {idx: i for i, idx in enumerate(train_unique_id)}
    id_maps['test'] = {idx: i for i, idx in enumerate(test_unique_id)}
    id_maps['query'] = id_maps['test']

    for mode in ['train', 'query', 'test']:
        out_list = []
        id_map = id_maps[mode]
        for file in files[mode]:
            if file[:2] == '-1':
                pid = i
            else:
                pid = id_map[int(file[:4])]
            cid = file[6]
            filename = os.path.join(img_root, mode , file)
            line = "{}\t{}\t{}\t{}".format(filename, pid, cid, did)
            out_list.append(line)
        write_list(out_list, os.path.join(root, name+'_'+mode+'.txt'))

def get_list(pic_list, id_map, domain, did):
    out_list = []
    for pics in pic_list:
        for pic in pics:
            pid = id_map[int(pic[6:6+5])]
            cid = pic[4]
            file_name = os.path.join(root, domain, pic)
            line = "{}\t{}\t{}\t{}".format(file_name, pid, cid, did)
            out_list.append(line)
    return out_list

def merge_source_txt(dataset_names):
    id_offset = 0
    out_list = []
    output_name = ""
    for dataset in dataset_names:
        output_name += dataset + '_'
    output_name += 'train.txt'

    for dataset in dataset_names:
        dataset_dir = root
        files, pids, cids, dids = read_kv(osp.join(dataset_dir, dataset + '_train.txt'))
        unique_ids = set(map(int, pids))
        id_mapping = {idx: i + id_offset for i, idx in enumerate(unique_ids)}
        for file, pid, cid, did in zip(files, pids, cids, dids):
            pid = id_mapping[int(pid)]
            line = "{}\t{}\t{}\t{}".format(file, pid, cid, did)
            out_list.append(line)
        id_offset += len(id_mapping)

    write_list(out_list, osp.join(root, output_name))
    print ("Max ID:", id_offset)

def read_list(file_path, coding=None):
    if coding is None:
        with open(file_path, 'r') as f:
            arr = [line.strip('\t') for line in f.readlines()]
    else:
        with codecs.open(file_path, 'r', coding) as f:
            arr = [line.strip('\t') for line in f.readlines()]
    return arr


def write_list(arr, file_path, coding=None):
    if coding is None:
        arr = ['{}'.format(item) for item in arr]
        with open(file_path, 'w') as f:
            f.write('\n'.join(arr))
    else:
        with codecs.open(file_path, 'w', coding) as f:
            f.write(u'\n'.join(arr))


def read_kv(file_path, coding=None):
    arr = read_list(file_path, coding)
    if len(arr) == 0:
        return [], []
    return zip(*map(str.split, arr))


def write_kv(k, v, file_path, coding=None):
    arr = zip(k, v)
    arr = [' '.join(item) for item in arr]
    write_list(arr, file_path, coding)


def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)


def write_list(arr, file_path, coding=None):
    if coding is None:
        arr = ['{}'.format(item) for item in arr]
        with open(file_path, 'w') as f:
            f.write('\n'.join(arr))
    else:
        with codecs.open(file_path, 'w', coding) as f:
            f.write(u'\n'.join(arr))

if __name__ == '__main__':
    #sources = ['cuhk03', 'viper', 'ilids', '3dpes']
    #for dataset in ['cuhk01', 'cuhk03', 'viper', 'ilids', '3dpes']:
    #    make_target_txt(dataset)
    #merge_source_txt(sources)
    make_market_txt(market_root, 'market')
    make_market_txt(duke_root, 'duke')