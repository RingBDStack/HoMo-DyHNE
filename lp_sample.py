import numpy as np
import random
datafile = 'dblp2'
dataset = 'dblp'
time_threshold = 0
max_time = 0
train_ratio = 0.8
if dataset == 'Aminer':  
    min_time = 1990
    max_time = 2005
elif dataset == 'dblp':
    min_time = 2000
    max_time = 2018
elif dataset == 'taobao':
    min_time = 2018120700
    max_time = 2018121223
time_threshold = int(min_time + (max_time-min_time)*train_ratio)
print('we choose edge of time below: ', time_threshold)
nodefile = './datasets/{}/{}.nodes'.format(datafile, dataset)
edgefile = './datasets/{}/{}.edges'.format(datafile, dataset)

out_file_train = './datasets/{}/{}.edges_lp_sample_train_{}'.format(
    datafile, dataset, train_ratio)
out_file_test = './datasets/{}/{}.edges_lp_sample_test_{}'.format(
    datafile, dataset, train_ratio)
out_file_neg = './datasets/{}/{}.edges_lp_sample_neg_{}'.format(datafile, dataset, train_ratio)
out_edge_list_train = './datasets/{}/{}.edgelist_lp_{}'.format(datafile, dataset, train_ratio)

nodes = []
edges = []
edges_train = []
edges_test = []
edge_dic = {}
degree = {}

with open(edgefile, 'r') as f:
    for line in f:
        line = line.strip().split()
        src, tgt, time = int(line[0]), int(line[1]), line[2]
        try:
            degree[src] += 1
        except:
            degree[src] = 1
        try:
            degree[tgt] += 1
        except:
            degree[tgt] = 1
        try:
            edge_dic[src].append(tgt)
        except:
            edge_dic[src] = [tgt]
        edges.append([src, tgt, time, 1])
print('dataset:{} has {} edges, {} nodes'.format(
    dataset, len(edges), len(degree)))
node_num = len(degree)
for eid, edge in enumerate(edges):
    t = int(edge[2]) if dataset!='taobao' else int(edge[2][0:10])
    src, tgt = edge[0], edge[1]
    d_src = degree[src]-1
    d_tgt = degree[tgt]-1
    if t >= time_threshold and d_src > 0 and d_tgt > 0:
        degree[src]-=1
        degree[tgt]-=1
        edges_test.append([src,tgt,edge[2]])
    else:
        edges_train.append([src,tgt,edge[2]])
assert len(edges_train)+len(edges_test) == len(edges)
# assert len(edges_test) == sample_num
print('train_edge_num:',len(edges_train))
print('test_edge_num:', len(edges_test))

neg_num = len(edges_test)
edges_neg = []
while neg_num > 0:
    a = random.randint(0, node_num-1)
    b = random.randint(0, node_num-1)
    if a == b:
        continue
    if a in edge_dic and b in edge_dic[a]:
        continue
    if b in edge_dic and a in edge_dic[b]:
        continue
    edges_neg.append([a, b, 0])
    neg_num -= 1
print('sample_edge_done')
print('neg_edge_num:',len(edges_neg))
isolated_num = 0
for node, degrees in degree.items():
    if degrees == 0:
        # edges_train.append([node, node, max_time]) add self-cycle
        degree[node]+=2
        isolated_num += 1
print('after_sample,isolated_node_num:{}'.format(isolated_num))
assert(len(edges_train)==len(edges)-len(edges_test)+isolated_num)
#check if there are still some isolated node
isolated_num = 0
for node, degrees in degree.items():
    if degrees == 0:
        isolated_num += 1
print('isolated_node_num:{}'.format(isolated_num))

print('out edge...')
with open(out_file_train, 'w') as fw:
    for edge in edges_train:
        fw.write('{}\t{}\t{}\n'.format(
            str(edge[0]), str(edge[1]), str(edge[2])))
with open(out_file_test, 'w') as fw:
    for edge in edges_test:
        fw.write('{}\t{}\t{}\n'.format(
            str(edge[0]), str(edge[1]), str(edge[2])))
with open(out_file_neg, 'w') as fw:
    for edge in edges_neg:
        fw.write('{}\t{}\t{}\n'.format(
            str(edge[0]), str(edge[1]), str(edge[2])))
with open(out_edge_list_train,'w') as fw:
    for edge in edges_train:
        fw.write('{}\t{}\n'.format(str(edge[0]),str(edge[1])))
print('done')
