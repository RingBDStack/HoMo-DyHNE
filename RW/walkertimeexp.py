import numpy as np
import networkx as nx
import random
import datetime
import math
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
import warnings
from queue import Queue


def find_far_neighbors(net, start, node, far_neighbors, visited, order):
    if order == k_order[net.nodes[start]['type']]:
        return

    for neighbor in net[node]:
        if neighbor in visited:
            continue

        if neighbor != start and net.nodes[neighbor]['type'] == net.nodes[start]['type']:
            far_neighbors.add(neighbor)
        find_far_neighbors(net, start, neighbor, far_neighbors, visited, order + 1)
        visited.append(neighbor)


def construct_net(filename='datasets/dblp/', dataset='dblp', isdirected=False):
    # if isdirected:
    #     net = nx.DiGraph()
    # else:
    #     net = nx.Graph()
    net = nx.MultiGraph()

    with open(filename + dataset + '.nodes') as f:
        for line in f:
            line = line.split()
            line = [int(e) for e in line]
            net.add_node(line[0], type=line[1],
                         trans=np.zeros(type_num), neighbors={}, neighbors_time={})
    print('nodes info loading is done!')
    print('node_num:',len(net.nodes()))
    print('edge_file:',filename + dataset + '.edges')
    degree=np.zeros(len(net.nodes()))
    with open(filename + dataset + '.edges') as f:
        min_time = 99999999999999
        max_time = -99999999999999
        for line in f:
            line = line.split()
            if (line[2] != 'always'):
                if (int(line[2]) > max_time):
                    max_time = int(line[2])
                if (int(line[2]) < min_time):
                    min_time = int(line[2])
            try:
                time = line[2]
            except:
                time = 'always'
            line = [int(e) for e in line[:2]]

            net.add_edge(line[0], line[1], time=time)
            degree[int(line[0])]+=1
            degree[int(line[1])]+=1
            # try:
            #     net.nodes[line[0]]['neighbors_time'][line[1]].append(time)
            # except:
            #     net.nodes[line[0]]['neighbors_time'][line[1]] = [time]
            # try:
            #     net.nodes[line[1]]['neighbors_time'][line[0]].append(time)
            # except:
            #     net.nodes[line[1]]['neighbors_time'][line[0]] = [time]
            try:
                net.nodes[line[0]]['neighbors_time'].append(
                    (line[1], time, net.nodes[line[1]]['type']))
            except:
                net.nodes[line[0]]['neighbors_time'] = [
                    (line[1], time, net.nodes[line[1]]['type'])]
            try:
                net.nodes[line[1]]['neighbors_time'].append(
                    (line[0], time, net.nodes[line[0]]['type']))
            except:
                net.nodes[line[1]]['neighbors_time'] = [
                    (line[0], time, net.nodes[line[0]]['type'])]
            # print(net.nodes[line[0]],type(line[0]),line[0])
    print('edges info loading is done!')
    
    for node in net.nodes():
        for neighbor in net[node]:
            try:
                net.nodes[node]['neighbors'][net.nodes[neighbor]
                ['type']].append(neighbor)
            except:
                net.nodes[node]['neighbors'][net.nodes[neighbor]
                ['type']] = [neighbor]
    # print(net.nodes[0])
    # print(net[0])
    print('neighbors info loading is done!')
    # for node in net.nodes():
    #     far_neighbors = set()
    #     find_far_neighbors(net, node, node, far_neighbors, [], 0)
    #     net.nodes[node]['far_neighbors'] = far_neighbors
    isolated_num = 0
    for node in net.nodes():
        if len(net.nodes[node]['neighbors_time']) == 0:
            isolated_num+=1
            # net.nodes[node]['neighbors_time']=[(0,'always',net.nodes[0]['type'])]
            # print(net.nodes[node]['neighbors'])    
            #net.nodes[node]['neighbors'] 
    print('isolated_node_num:',isolated_num)
    return net, min_time, max_time, degree


def unigram_sample(population, size=1, replace=True, weight=None):
    weight_sum = np.sum(weight)
    weight = weight / weight_sum

    # if weight_sum != 1:
    # weight = weight / weight_sum
    # weight_sum = np.sum(weight)
    # if weight_sum != 1:
    #     weight = weight / weight_sum
    #     weight_sum = np.sum(weight)
    #     if weight_sum != 1:
    #         weight = weight / weight_sum
    return np.random.choice(population, size=size, replace=replace, p=weight)


def clear_trans():
    for node in net.nodes():
        for type_ in net.nodes[node]['neighbors']:
            net.nodes[node]['trans'][type_] = 1


def walk(arg_list=(1, 0)):
    type_num = 0
    with open(filename + dataset + '.node_types') as f:
        for line in f:
            type_num += 1
    times = 0
    walks = []
    walk_times = arg_list[0]
    process_id = arg_list[1]
    # clear_trans()
    print(process_id, ':', 'start walking...', len(
        net.nodes()), datetime.datetime.now())
    for _ in range(walk_times):
        for root_node in net.nodes():
            # clear_trans()
            walk_instance = []

            local_trans = {}
            for node in net.nodes():
                local_trans[node] = np.zeros(type_num)
                for type_ in net.nodes[node]['neighbors']:
                    local_trans[node][type_] = 1

            cur_node = root_node
            walk_instance.append(cur_node)
            for t in range(walk_length):

                temp = [local_trans[cur_node]]
                for node in net.nodes[cur_node]['far_neighbors']:
                    temp.append(local_trans[node])
                local_trans[cur_node] = np.mean(temp, axis=0)
                for i, each in enumerate(local_trans[cur_node]):
                    if i not in net.nodes[cur_node]['neighbors']:
                        local_trans[cur_node][i] = 0

                cur_type = unigram_sample(population=range(
                    type_num), size=1, replace=True, weight=local_trans[cur_node])[0]
                local_trans[cur_node][cur_type] += 1
                cur_node = random.choice(net.nodes[cur_node]['neighbors'][cur_type])
                walk_instance.append(cur_node)
            walks.append(walk_instance)
            times += 1
            if times % 1000 == 0:
                print(process_id, ':', times, '/', len(net.nodes()),
                      datetime.datetime.now())
    return walks


def walk_time(arg_list=(1, 0)):
    type_num = 0
    with open(filename + dataset + '.node_types') as f:
        for line in f:
            type_num += 1
    net = arg_list[2]
    max_time = arg_list[3]
    min_time = arg_list[4]
    times = 0
    walks = []
    walk_times = []
    global_trans = {}
    root_nodes = arg_list[0]  # root_nodes[process_id]
    process_id = arg_list[1]
    # max_degree = 0
    # clear_trans()
    print(process_id, ':', 'start walking...', len(
        root_nodes), datetime.datetime.now())
    # print(net.nodes)
    for root_node in root_nodes:
        # clear_trans()
        if(degree[root_node]==0):
            continue
        pre_time = str(min_time)
        walk_instance = []

        local_trans = {}

        flow = {}
        for type_ in range(type_num):
            if k == -1:
                flow[type_] = np.zeros(type_num)
            else:
                flow[type_] = Queue(maxsize=k)

        cur_node = root_node
        walk_instance.append(cur_node)
        time_path = []  # cur_node , time, time, ...
        time_path.append(cur_node)
        for t in range(walk_length):

            if cur_node not in local_trans:
                local_trans[cur_node] = np.zeros(type_num)
                for type_ in net.nodes[cur_node]['neighbors']:
                    local_trans[cur_node][type_] = 1

            if cur_node not in global_trans:
                global_trans[cur_node] = np.zeros(type_num)
                for type_ in net.nodes[cur_node]['neighbors']:
                    global_trans[cur_node][type_] = 1

            temp = {}
            if k == -1:
                for type_ in range(type_num):
                    temp[type_] = 1
                    if local_trans[cur_node][type_] == 0:
                        continue
                    local_trans[cur_node][type_] += flow[net.nodes[cur_node]
                    ['type']][type_]
                    global_trans[cur_node][type_] += flow[net.nodes[cur_node]
                    ['type']][type_]
                    temp[type_] += 1
            else:
                for each in flow[net.nodes[cur_node]['type']].queue:
                    for type_ in range(type_num):
                        temp[type_] = 1
                        if local_trans[cur_node][type_] == 0:
                            continue
                        local_trans[cur_node][type_] += each[type_]
                        global_trans[cur_node][type_] += each[type_]
                        temp[type_] += 1
            for type_, v in temp.items():
                local_trans[cur_node][type_] /= v

            if random.random() < alpha:
                next_type = unigram_sample(population=range(
                    type_num), size=1, replace=True, weight=local_trans[cur_node])[0]

            else:
                next_type = random.choice(
                    list(net.nodes[cur_node]['neighbors'].keys()))

            neighbors_t = [e for e in net.nodes[cur_node]['neighbors_time'] if
                           (e[1] == 'always' or e[1] >= pre_time) and e[2] == next_type]
            if len(neighbors_t) == 0:
                neighbors_t = [e for e in net.nodes[cur_node]['neighbors_time'] if
                               (e[1] == 'always' or e[1] >= pre_time)]
            time_weight = []
            for e in neighbors_t:
                e_time = int(e[1]) if e[1] != 'always' else max_time
                time_weight.append(math.exp(delta * (e_time - int(pre_time))))
            # if len(time_weight) > max_degree:
            #     max_degree = len(time_weight)
            index = unigram_sample(population=range(len(neighbors_t)), size=1, replace=False, weight=time_weight)

            next_node, now_time, _ = neighbors_t[index[0]]

            if now_time != 'always':
                pre_time = now_time

            local_trans[cur_node][next_type] += 1
            global_trans[cur_node][next_type] += 1
            
            if k == -1:
                flow[net.nodes[cur_node]['type']][next_type] += 1
            else:
                if flow[net.nodes[cur_node]['type']].full():
                    flow[net.nodes[cur_node]['type']].get()
                flow[net.nodes[cur_node]['type']].put(local_trans[cur_node])

            cur_node = next_node
            time_path.append(pre_time)
            walk_instance.append(cur_node)

        walks.append(walk_instance)
        walk_times.append(time_path)
        times += 1
        if times % 1000 == 0:
            print(process_id, ':', times, '/', len(root_nodes),
                  datetime.datetime.now())
            print(len(time_path), time_path)
    # print('max_degree:', max_degree)
    print(process_id, ':', 'done!')
    return walks, global_trans, walk_times


def walk_normal(arg_list=(1, 0)):
    type_num = 0
    with open(filename + dataset + '.node_types') as f:
        for line in f:
            type_num += 1
    times = 0
    walks = []
    root_nodes = arg_list[0]
    process_id = arg_list[1]
    # clear_trans()
    print(process_id, ':', 'start walking...', len(
        root_nodes), datetime.datetime.now())
    for root_node in root_nodes:
        # clear_trans()
        walk_instance = []

        cur_node = root_node
        walk_instance.append(cur_node)
        for t in range(walk_length):
            #     cur_type = random.choice(list(net.nodes[cur_node]['neighbors'].keys()))
            #     cur_node = random.choice(net.nodes[cur_node]['neighbors'][cur_type])
            cur_node = random.choice(list(net[cur_node]))
            walk_instance.append(cur_node)
        walks.append(walk_instance)
        times += 1
        if times % 1000 == 0:
            print(process_id, ':', times, '/', len(root_nodes),
                  datetime.datetime.now())
    print(process_id, ':', 'done!')
    return walks


def walk_multiprocess(walk_times=10, max_num_workers=cpu_count()):
    """ Use multi-process scheduling"""
    # allocate walk times to workers
    print("walk_times:{},walk_legnth:{}".format(walk_times, walk_length))
    div, mod = divmod(walk_times * len(net.nodes), max_num_workers)
    times_per_worker = [div for _ in range(max_num_workers)]
    for idx in range(mod):
        times_per_worker[idx] = times_per_worker[idx] + 1

    count = 0
    idx = 0
    root_nodes = [[] for _ in range(max_num_workers)]
    for _ in range(walk_times):
        for n in net.nodes():
            root_nodes[idx].append(n)
            count += 1
            if count == times_per_worker[idx]:
                idx += 1
                count = 0
    print(times_per_worker)

    sens = []
    total_trans = []
    args_list = []
    time_sens = []
    for index in range(len(root_nodes)):
        args_list.append((root_nodes[index], index, net, max_time,
                          min_time))  # ([node1,node2,...](len is times_per_worker[index]),index)
    start_time = datetime.datetime.now()
    print('Start! ', start_time)
    with ProcessPoolExecutor(max_workers=max_num_workers) as executor:
        for walks, trans, times in executor.map(walk_time, args_list):
            sens.extend(walks)
            total_trans.append(trans)
            time_sens.extend(times)

    end_time = datetime.datetime.now()
    print('Done!', datetime.datetime.now())
    print('Time cost:', end_time-start_time)

    with open(out_filename, 'w') as f:
        for walk_instance in sens:
            for n in walk_instance:
                f.write(u"{} ".format(str(n)))
            f.write('\n')

    final_trans = {}
    for trans in total_trans:
        for node in trans:
            try:
                final_trans[node]
            except:
                final_trans[node] = np.zeros(type_num)
            final_trans[node] += trans[node]

    for node in final_trans:
        final_trans[node] /= np.sum(final_trans[node])

    with open(out_trans_filename, 'w') as f:
        for node, trans in final_trans.items():
            f.write(u"{} ".format(str(node)))
            for n in trans:
                f.write(u"{} ".format(str(n)))
            f.write('\n')
    with open(out_time_file, 'w') as f:
        for time in time_sens:
            for x in time:
                f.write(u"{} ".format(str(x)))
            f.write('\n')


filename = './datasets/taobao_50000/'
dataset = 'taobao'
repeated_order = 5

walk_length = 10
walk_times = 40
delta = -1e-5
delta_str = '1e-5'
# out_filename = dataset + '/{}.time.10.40.walkexp.1_{}'.format(dataset,repeated_order)
# out_trans_filename = dataset + '/{}.time.10.40.transexp.1_{}'.format(dataset,repeated_order)
# out_time_file = dataset + '/{}.time.10.40.timesexp.1_{}'.format(dataset,repeated_order)
# out_standard_time_file = dataset + '/{}.time.{}.{}.timesexp.1.normalization_{}'.format(dataset,walk_length, walk_times, repeated_order)

# out_filename = dataset + '/{}.time.{}.{}.walkexp.{}'.format(dataset, walk_length, walk_times, delta_str)
# out_trans_filename = dataset + '/{}.time.{}.{}.transexp.{}'.format(dataset,walk_length, walk_times, delta_str)
# out_time_file = dataset + '/{}.time.{}.{}.timesexp.{}'.format(dataset,walk_length, walk_times, delta_str)
# out_standard_time_file = dataset + '/{}.time.{}.{}.timesexp.{}.normalization'.format(dataset,walk_length, walk_times, delta_str)

out_filename = dataset + '/{}.time.{}.{}.walkexp.{}_{}'.format(dataset, walk_length, walk_times, delta_str, repeated_order)
out_trans_filename = dataset + '/{}.time.{}.{}.transexp.{}_{}'.format(dataset,walk_length, walk_times, delta_str, repeated_order)
out_time_file = dataset + '/{}.time.{}.{}.timesexp.{}_{}'.format(dataset,walk_length, walk_times, delta_str, repeated_order)
out_standard_time_file = dataset + '/{}.time.{}.{}.timesexp.{}.normalization_{}'.format(dataset,walk_length, walk_times, delta_str, repeated_order)

# out_filename = 'test.walk'
# out_trans_filename = 'test.trans'
f = open(out_filename, 'w')
f.truncate()
f.close()
f = open(out_trans_filename, 'w')
f.truncate()
f.close()
# wl=10,20,40,80,160,320,640

#
# # wt=1,5,10,20,40,80,160
print('delta:',delta)
print('dataset:',dataset)
type_num = 0
k_order = {}
k = 5
alpha = 0.8

if __name__ == "__main__":
    print('cpu_num:', cpu_count())
    meta_schema = nx.Graph()
    with open(filename + dataset + '.edge_types') as f:
        for line in f:
            line = line.split()
            line = [int(e) for e in line]
            meta_schema.add_node(line[0])
            meta_schema.add_node(line[1])
            meta_schema.add_edge(line[0], line[1])

    type_num = len(meta_schema.nodes())
    for type_ in meta_schema:
        if type_ in meta_schema[type_]:
            k_order[type_] = 1
        else:
            k_order[type_] = 2

    net, min_time, max_time,degree = construct_net(filename=filename, dataset=dataset, isdirected=False)
    
    print('min_time', min_time)
    print('max_time', max_time)

    walk_multiprocess(walk_times=walk_times)  # , max_num_workers=1)
    #standard time
    max_min = max_time - min_time
    print('standard_time begin!')
    with open(out_time_file, 'r') as f:
        with open(out_standard_time_file,'w') as fw:
            for line in f:
                line = line.strip().split()
                id = int(line[0])
                fw.write('{} '.format(str(id)))
                for t in line[1:]:
                    time = int(t)
                    time=(time-min_time)/max_min
                    fw.write('{} '.format(str(time)))
                fw.write('\n')
    print('standard_time done!')

