from math import e
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import random
import datetime
import argparse
import pickle

def eval_clf(emb_dic, label):
    X = []
    Y = []
    for p in label:
        if p not in emb_dic:
            continue
        X.append(emb_dic[p])
        Y.append(label[p])
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)
    L = LogisticRegression()
    L.fit(X_train, Y_train)
    Y_pred = L.predict(X_test)
    Micro_F1 = f1_score(Y_test, Y_pred, average='micro')
    Macro_F1 = f1_score(Y_test, Y_pred, average='macro')
    Acc = sum(Y_pred[i] == Y_test[i] for i in range(len(Y_test))) / len(Y_test)
    print('Micro_F1 = {:.4f} , Macro_F1 = {:.4f} , ACC = {:.4f}'.format(
        Micro_F1, Macro_F1, Acc))
    return Micro_F1, Macro_F1, Acc


def eval_nmi(emb_dic, label, n_label):
    X = []
    Y = []
    for p in label:
        if p not in emb_dic:
            continue
        X.append(emb_dic[p])
        Y.append(label[p])
    cluster = KMeans(n_label, random_state=0)
    cluster.fit(X)
    Y_pred = cluster.predict(X)
    nmi = normalized_mutual_info_score(Y, Y_pred)
    print('NMI = {:.4f}'.format(nmi))
    return nmi


def link_prediction(emb_dic, train_ratio, pos_edge, neg_edge, op):
    print('train_ratio={}, op={}'.format(train_ratio, op))
    edges_all = pos_edge + neg_edge
    label_all = [1]*len(pos_edge) + [0]*len(neg_edge)
    edges_train, edges_test, labels_train, labels_test = train_test_split(
        edges_all, label_all, test_size=1-train_ratio, random_state=0, shuffle=True, stratify=label_all)
    train1 = np.array([emb_dic[e[0]]for e in edges_train], dtype=float)
    train2 = np.array([emb_dic[e[1]]for e in edges_train], dtype=float)
    test1 = np.array([emb_dic[e[0]]for e in edges_test], dtype=float)
    test2 = np.array([emb_dic[e[1]]for e in edges_test], dtype=float)
    if op == 'l1':
        X_train = np.abs(train1-train2)
        X_test = np.abs(test1-test2)
    elif op == 'l2':
        X_train = np.square(train1-train2)
        X_test = np.square(test1-test2)
    elif op == 'hadamard':
        X_train = np.multiply(train1, train2)
        X_test = np.multiply(test1, test2)
    elif op == 'average':
        X_train = (train1+train2)/2
        X_test = (test1+test2)/2
    else:
        print('invalid feature operation: {}'.format(op))
    LR = LogisticRegression()
    LR.fit(X_train, np.array(labels_train))
    pred = LR.predict(X_test)
    acc = accuracy_score(labels_test, pred)
    ap = average_precision_score(labels_test, pred)
    auc = roc_auc_score(labels_test, pred)
    f1 = f1_score(labels_test, pred)
    return acc, ap, auc, f1


def init_args():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('-d', '--dataset', default='Aminer',
                        type=str, help='dataset for eval')
    parser.add_argument('-m', '--model', default='htne', type=str)
    parser.add_argument('-div', '--divide_num', default=100, type=int)
    parser.add_argument('-o', '--op', default='l1', type=str)
    parser.add_argument('-tr', '--train_ratio', default=0.5, type=float)
    parser.add_argument('-tir','--time_ratio',default=0.8, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('begin test!')
    args = init_args()
    warnings.filterwarnings('ignore')
    if args.dataset == 'Aminer':
        node_file = './datasets/aminer2/Aminer.nodes'
        label_file = './datasets/aminer2/Aminer.labels'
        edge_file = './datasets/aminer2/Aminer.edges'
        # sample_lp_edge_file = f'./datasets/aminer2/Aminer.edges_lp_sample_test3'
        # sample_lp_edge_neg_file = f'./datasets/aminer2/Aminer.edges_lp_sample_neg3'
        sample_lp_edge_file = f'./datasets/aminer2/Aminer.edges_lp_sample_test_{args.time_ratio}'
        sample_lp_edge_neg_file = f'./datasets/aminer2/Aminer.edges_lp_sample_neg_{args.time_ratio}'
    elif args.dataset == 'dblp':
        node_file = './datasets/dblp2/dblp.nodes'
        label_file = './datasets/dblp2/dblp.labels'
        edge_file = './datasets/dblp2/dblp.edges'
        sample_lp_edge_file = f'./datasets/dblp2/dblp.edges_lp_sample_test_{args.time_ratio}'
        sample_lp_edge_neg_file = f'./datasets/dblp2/dblp.edges_lp_sample_neg_{args.time_ratio}'
    elif args.dataset == 'taobao':
        node_file = './datasets/taobao_50000/taobao.nodes'
        label_file = './datasets/taobao_50000/taobao.labels'
        edge_file = './datasets/taobao_50000/taobao.edges'
        sample_lp_edge_file = f'./datasets/taobao_50000/taobao.edges_lp_sample_test_{args.time_ratio}'
        sample_lp_edge_neg_file = f'./datasets/taobao_50000/taobao.edges_lp_sample_neg_{args.time_ratio}'
    else:
        print('invalid dataset!')
    print(f'sample_lp_edge_file:{sample_lp_edge_file}')
    print(f'sample_lp_edge_neg_file:{sample_lp_edge_neg_file}')
    repeated_time = 5
    emb_dic = {}
    label_dic = {}
    edges = []
    lp_pos_edges = []
    lp_neg_edges = []
    n_label = 1
    op = args.op

    micro_list = []
    macro_list = []
    nmi_list = []
    acc_list = []
    ap_list = []
    auc_list = []
    f1_list = []
    print('load label file...')
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            id, label = int(line[0]), int(line[1])
            if label not in label_dic.values():
                n_label += 1
            label_dic[id] = label
    # print('load edge file...')
    # with open(edge_file, 'r') as f:
    #     for line in f:
    #         line = line.strip().split()
    #         source = int(line[0])
    #         target = int(line[1])
    #         edges.append((source, target))
    with open(sample_lp_edge_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            src, tgt = int(line[0]), int(line[1])
            lp_pos_edges.append((src, tgt))
    with open(sample_lp_edge_neg_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            src, tgt = int(line[0]), int(line[1])
            lp_neg_edges.append((src, tgt))
    print('load file done：', datetime.datetime.now())
    # edge_num_sample_has_in_edges = sum(
    #     sample_edges[i] in edges for i in range(len(sample_edges)))
    print('eval_model:', args.model)
    print('dataset:', args.dataset)
    print('train_ratio:', args.train_ratio)
    print('op:', args.op)
    print('time ratio:',args.time_ratio)
    print('pos_edge_num:', len(lp_pos_edges))
    print('neg_edge_num:', len(lp_neg_edges))
    # print('there are {} edges in real edges for sample_edges {}'.format(edge_num_sample_has_in_edges,datetime.datetime.now()))
# evalutaion
    for i in range(repeated_time):
        # load emb_file
        emb_file = ''
        if args.model == 'htne' or args.model == 'MMDNE' or args.model == 'MTNE':
            if args.model == 'htne':
                emb_file = './emb/HTNE/' + args.dataset + \
                    f'_htne_attn_lp_{args.time_ratio}_50.emb_' + str(i + 1) if args.dataset!='dblp' else './emb/HTNE/' + args.dataset + \
                    f'_htne_attn_lp_{args.time_ratio}_30.emb_' + str(i + 1)
            elif args.model == 'MMDNE':
                emb_file = './emb/MMDNE/' + args.dataset + \
                    f'.tne_epoch30_lr0.02_his5_neg5_eps0.4.lp_{args.time_ratio}.emb_' + str(i + 1)
            else:
                emb_file = './emb/MTNE/{}_mtne_attn_30.lp_{}.emb_{}'.format(
                    args.dataset, args.time_ratio,str(i+1))
            id = 0
            with open(emb_file, 'r') as f:
                f.readline()
                for line in f:
                    line = line.strip().split()
                    emb = [float(x) for x in line[0:]]
                    emb_dic[id] = emb
                    id += 1
        elif args.model == 'FSG_FLOW' or args.model == 'deepwalk' \
            or args.model == 'LINE' or args.model == 'node2vec' \
                or args.model == 'FSG_TIME' or args.model == 'HFSG':
            if args.model == 'deepwalk':
                emb_file = './emb/deepwalk/' + args.dataset + \
                    f'.deepwalk.10.40.lp.{args.time_ratio}.embeddings_' + str(i + 1)
            elif args.model == 'node2vec':
                emb_file = './emb/node2vec/{}.node2vec.10.40.lp.{}.embeddings_{}'.format(
                    args.dataset,args.time_ratio, str(i+1))
            elif args.model == 'LINE':
                emb_file = './emb/LINE/' + args.dataset + \
                    f'.line.lp.{args.time_ratio}.txt_' + str(i + 1)
            # elif args.model == 'FSG_TIME':
            #     emb_file = './emb/FSGNS/time/{}.FSGNS.timeexp.1.lp.10.40.3.embedding.txt_{}'.format(
            #         args.dataset, str(i+1))
            #     if args.dataset == 'taobao':
            #         emb_file = './emb/FSGNS/time/{}.FSGNS.timeexp.1e-6.lp.10.40.4.embedding.txt_{}'.format(
            #             args.dataset, str(i+1))+'_epoch_10'
            elif args.model == 'HFSG':
                if args.dataset == 'Aminer':
                    emb_file='./emb/HFSGNS/{}.HFSGNS.time.lp.10.40.3.embedding.exp.1_{}.txt_{}'.format(args.dataset,args.time_ratio,str(i+1))
                if args.dataset == 'dblp':
                    emb_file='./emb/HFSGNS/{}.HFSGNS.time.lp.10.40.3.embedding.exp.0.5_{}.txt_{}'.format(args.dataset,args.time_ratio,str(i+1))
                elif args.dataset == 'taobao':
                    emb_file = './emb/HFSGNS/{}.HFSGNS.time.lp.10.40.4.embedding.exp.1e-6_{}.txt_{}'.format(
                        args.dataset,args.time_ratio, str(i+1))
            else:
                print('input legal model!')
            try:
                with open(emb_file, 'r') as f:
                # print(f.readline())
                    for line in f:
                        line = line.strip().split()
                        id = int(line[0])
                        emb = [float(x) for x in line[1:]]
                        emb_dic[id] = emb
            except:
                emb_file = emb_file + '_epoch_10'
                with open(emb_file, 'r') as f:
                # print(f.readline())
                    for line in f:
                        line = line.strip().split()
                        id = int(line[0])
                        emb = [float(x) for x in line[1:]]
                        emb_dic[id] = emb
        # elif args.model == 'TGAT' or args.model == 'CAW':
        #     if args.model == 'CAW':
        #         emb_file = './emb/CAW/CAW.{}.embeddings'.format(args.dataset)
        #     elif args.model == 'TGAT':
        #         emb_file = './emb/TGAT/TGAT.{}.embeddings'.format(args.dataset)
        #     with open(emb_file, 'r') as f:
        #         for line in f:
        #             line = line.strip().split()
        #             id = int(line[0])-1
        #             emb = [float(x) for x in line[1:]]
        #             if id in emb_dic:
        #                 print('duplicated!')
        #             emb_dic[id] = emb
        elif args.model == 'Metapath2vec' or args.model == 'MetaGraph2vec' or args.model=='JUST':
            name = 'taobao_50000' if args.dataset == 'taobao' else 'aminer2' if args.dataset == 'Aminer' else 'dblp2'
            meta_s = 'uitiu' if args.dataset == 'taobao' else 'apcpa'
            if args.model == 'Metapath2vec':
                emb_file = './OpenHINE-master/output/embedding/Metapath2vec/{}_{}.lp_{}.txt_{}'.format(
                    name, meta_s, args.time_ratio, str(i+1))
            elif args.model == 'JUST':
                emb_file = f'./emb/JUST/{args.dataset}.lp.{args.time_ratio}.embeddings_{str(i+1)}'
            else:
                emb_file = './OpenHINE-master/output/embedding/MetaGraph2vec/{}_{}_node.lp_{}.txt_{}'.format(
                    name, meta_s, args.time_ratio, str(i+1))
            with open(emb_file, 'r') as f:
                f.readline()
                for line in f:
                    line = line.strip().split()
                    # print(line[0])
                    try:
                        id = int(line[0][1:])
                    except:
                        continue
                        print(line[0][1:])
                    emb = [float(x) for x in line[1:]]
                    emb_dic[id] = emb
        elif args.model == 'CTDNE':
            emb_file = f'./emb/CTDNE/{args.dataset}.10.40.lp.{args.time_ratio}.w2v_{str(i+1)}.pkl'
            with open(emb_file, 'rb')as f:
                embeddings = pickle.load(f)
                max_id = 0
                # print(embeddings)
                for key,val in embeddings.items():
                    id = int(key)
                    max_id = id if id > max_id else max_id
                    emb = val.tolist()
                    emb_dic[id] = emb
        print('node_num:',len(emb_dic))
        print('emb_file:',emb_file)
        acc, ap, auc, f1 = link_prediction(
            emb_dic, args.train_ratio, lp_pos_edges, lp_neg_edges, args.op)
        acc_list.append(acc)
        ap_list.append(ap)
        auc_list.append(auc)
        f1_list.append(f1)
    
    print('acc_list:{}\nap_list:{}\nauc_list:{}\nf1_list:{}'.format(
        acc_list, ap_list, auc_list, f1_list))
    print('acc: {:.4f}$\pm${:.4f}'.format(np.mean(acc_list), np.std(acc_list)))
    print('auc: {:.4f}$\pm${:.4f}'.format(np.mean(auc_list), np.std(auc_list)))
    print('ap: {:.4f}$\pm${:.4f}'.format(np.mean(ap_list), np.std(ap_list)))
    print('f1: {:.4f}$\pm${:.4f}'.format(np.mean(f1_list), np.std(f1_list)))