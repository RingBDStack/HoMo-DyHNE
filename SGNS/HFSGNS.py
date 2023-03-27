import numpy as np
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import init
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import datetime
import time

FType = torch.FloatTensor
DID = 1
debug = 1


# from torch.utils.tensorboard import SummaryWriter

# # default 'log_dir' is 'runs'
# write = SummaryWriter("runs/FSGNS")


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputTimeFileName, inputFileName, feature_dimension, featureFileName, labelFileName, use_fea,
                 min_count, care_type,
                 trans_file_name, trans_dim, emb_dimension=128, set_normal=0):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.use_fea = use_fea
        self.feature = []
        self.trans = []
        self.word2label = {}
        self.labels = set()

        self.emb_dimension = emb_dimension
        self.set_normal = set_normal
        print('use normalization:', self.set_normal)
        self.inputFileName = inputFileName
        self.inputTimeFileName = inputTimeFileName
        self.featureFileName = featureFileName
        self.read_words(min_count)
        self.read_features(use_fea, feature_dimension, featureFileName, trans_file_name, trans_dim)
        self.read_labels(labelFileName)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName):
            line = line.split()
            # for line in data.split('\n'):
            #     line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print(
                                "Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            # if c < min_count:
            #     continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        with open(self.featureFileName, 'r') as f:
            for line in f:
                line = line.strip().split()
                w = line[0]
                if w not in self.word2id:
                    self.word2id[w] = wid
                    self.id2word[wid] = w
                    wid += 1
        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def read_features(self, use_fea, feature_dimension, feature_file_name, trans_file_name, trans_dim):
        if use_fea == 'mix':
            init_range = 1.0 / np.sqrt(self.word_count)
            # init_range = 1.0 / 128
            features = np.zeros([len(self.word2id), feature_dimension])
            with open(feature_file_name) as f:
                for line in f:
                    line = line.split()
                    feat = np.array([float(e) for e in line[1:]])
                    if self.set_normal == 1:
                        min = np.min(feat)
                        max = np.max(feat)
                        feat = -init_range + (feat - min) / (max - min) * (2 * init_range)
                    features[[self.word2id[line[0]]], :] = feat
            features = torch.FloatTensor(features)

            trans_matrix = np.zeros([len(self.word2id), trans_dim])
            with open(trans_file_name) as f:
                for line in f:
                    line = line.split()
                    trans = np.array([float(e) for e in line[1:]])
                    if self.set_normal == 1:
                        trans = -init_range + trans * (2 * init_range)  # min = 0 max = 1
                    trans_matrix[[self.word2id[line[0]]], :] = trans
            trans_matrix = torch.FloatTensor(trans_matrix)

            self.feature = torch.cat([features, trans_matrix], dim=1)
        elif use_fea == 'all':
            features = np.zeros([len(self.word2id), feature_dimension])
            with open(feature_file_name) as f:
                for line in f:
                    line = line.split()
                    features[[self.word2id[line[0]]], :] = np.array(
                        [float(e) for e in line[1:]])
            features = torch.FloatTensor(features)
            self.feature = torch.FloatTensor(features)

            trans_matrix = np.zeros([len(self.word2id), trans_dim])
            with open(trans_file_name) as f:
                for line in f:
                    line = line.split()
                    trans_matrix[[self.word2id[line[0]]], :] = np.array(
                        [float(e) for e in line[1:]])
            trans_matrix = torch.FloatTensor(trans_matrix)
            self.trans = torch.FloatTensor(trans_matrix)


        elif use_fea == 'fea':
            features = np.zeros([len(self.word2id), feature_dimension])
            with open(feature_file_name) as f:
                for line in f:
                    line = line.split()
                    features[[self.word2id[line[0]]], :] = np.array(
                        [float(e) for e in line[1:]])
            self.feature = torch.FloatTensor(features)

        elif use_fea == 'one-hot':
            # features = np.zeros([len(self.word2id), len(self.word2id)])
            # for i in range(len(self.word2id)):
            #     features[i][i] = 1
            # self.feature = torch.FloatTensor(features)
            self.feature = torch.tensor([[1, 0], [0, 1]])

        elif use_fea == 'one-hot-vec':
            features = np.zeros([len(self.word2id), len(self.word2id)])
            for i in range(len(self.word2id)):
                features[i][i] = 1
            self.feature = torch.FloatTensor(features)

        elif use_fea == 'trans':
            trans_matrix = np.zeros([len(self.word2id), trans_dim])
            with open(trans_file_name) as f:
                for line in f:
                    line = line.split()
                    trans_matrix[[self.word2id[line[0]]], :] = np.array(
                        [float(e) for e in line[1:]])
            self.feature = torch.FloatTensor(trans_matrix)
        print('Features:', self.feature.shape[1])
        # elif use_fea == 'adj':
        #     adj = np.zeros((node_num, node_num))
        #     with open(edge_file_name) as f:
        #         for line in f:
        #             line = line.split()
        #             i = int(line[0])
        #             j = int(line[1])
        #             adj[i][j] = 1
        #             adj[j][i] = 1
        #     # feature = np.dot(adj, np.random.rand(node_num, feature_dimension))
        #     self.feature = adj

    def read_labels(self, labelFileName):
        with open(labelFileName) as f:
            for line in f:
                line = line.split()
                self.word2label[line[0]] = int(line[1])
                self.labels.add(line[1])

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class RandomWalkDataset(Dataset):
    def __init__(self, data, window_size, neg_num):
        # read in dataset, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="ISO-8859-1")
        self.input_time_file = open(data.inputTimeFileName, encoding="ISO-8859-1")
        self.neg_num = neg_num
        self.his_len = 5

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            time_line = self.input_time_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()
            if not time_line:
                self.input_time_file.seek(0, 0)
                time_line = self.input_time_file.readline()
            if len(line) > 1:
                id2time = {}
                words = line.split()
                times = time_line.split()[1:]
                for index, word in enumerate(words[1:]):
                    id2time[self.data.word2id[word]] = float(times[index])
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]
                    # and np.random.rand() < self.dataset.discards[self.dataset.word2id[w]]
                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        # TODO 检查下界能否为0，因为第一个节点的时间信息是缺失的
                        # window_size > his_len
                        word_in_windows = word_ids[max(i - self.window_size, 1):i] + word_ids[
                                                                                     i + 1:i + self.window_size]
                        for j, v in enumerate(word_in_windows):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            v_time = id2time[v]

                            v_his = np.zeros((self.his_len))
                            his_nodes = word_in_windows[max(j - self.his_len, 0):j]
                            v_his[:len(his_nodes)] = his_nodes

                            v_his_time = np.zeros(self.his_len)
                            his_time = [id2time[his] for his in his_nodes]
                            v_his_time[:len(his_time)] = his_time

                            pair_catch.append(
                                (u, v, self.data.getNegatives(v, self.neg_num), v_time, v_his, v_his_time))
                    return pair_catch

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _, _, _, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _,
                                                    _, neg_v, _, _, _ in batch if len(batch) > 0]
        all_v_time = [v_time for batch in batches for _, _, _, v_time, _, _ in batch if len(batch) > 0]
        all_v_his = [v_his for batch in batches for _, _, _, _, v_his, _ in batch if len(batch) > 0]
        all_v_his_time = [v_his_time for batch in batches for _, _, _, _, _, v_his_time in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v), \
               torch.LongTensor(all_v_time), \
               torch.LongTensor(all_v_his), \
               torch.LongTensor(all_v_his_time)


class FeatureSkipGramModel(nn.Module):

    def __init__(self, feature_dimension, emb_dimension, neg_k, use_fea, emb_size, trans_dimension, set_normal, act_func):
        super(FeatureSkipGramModel, self).__init__()
        self.feature_dimension = feature_dimension
        self.emb_dimension = emb_dimension
        self.trans_dimension = trans_dimension
        self.neg_k = neg_k
        self.use_fea = use_fea
        self.emb_size = emb_size  # node_num
        self.debug = True
        self.set_normal = set_normal
        self.act_func = act_func
        if use_fea == 'one-hot':
            self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
            self.v_embeddings = nn.Embedding(emb_size, emb_dimension)

            initrange = 1.0 / self.emb_dimension
            init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
            init.constant_(self.v_embeddings.weight.data, 0)
        elif use_fea == 'mix':
            # self.node_emb = nn.Parameter(torch.from_numpy(np.random.uniform(
            #     -1. / np.sqrt(self.emb_size), 1. / np.sqrt(self.emb_size), (self.emb_size, emb_dimension))).type(
            #     FType))

            self.delta = nn.Parameter((torch.zeros(self.emb_size) + 1.).type(FType))

            self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
            self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
            if self.set_normal == 1:
                initrange = 1.0 / np.sqrt(self.emb_size)
            else:
                initrange = 1.0 / emb_dimension
            init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
            init.constant_(self.v_embeddings.weight.data, 0)

            self.Wu1 = nn.Parameter(torch.empty(size=(feature_dimension, emb_dimension)))
            self.Wv1 = nn.Parameter(torch.empty(size=(feature_dimension, emb_dimension)))
            self.Wu2 = nn.Parameter(torch.empty(size=(2 * emb_dimension, emb_dimension)))
            self.Wv2 = nn.Parameter(torch.empty(size=(2 * emb_dimension, emb_dimension)))

            nn.init.xavier_uniform_(self.Wu1.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wv1.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wu2.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wv2.data, gain=1.414)

        elif use_fea == 'all':
            self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
            self.v_embeddings = nn.Embedding(emb_size, emb_dimension)

            initrange = 1.0 / self.emb_dimension
            init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
            init.constant_(self.v_embeddings.weight.data, 0)

            self.Wuf = nn.Parameter(torch.empty(size=(feature_dimension, emb_dimension)))
            self.Wvf = nn.Parameter(torch.empty(size=(feature_dimension, emb_dimension)))
            self.Wut = nn.Parameter(torch.empty(size=(trans_dimension, emb_dimension)))
            self.Wvt = nn.Parameter(torch.empty(size=(trans_dimension, emb_dimension)))
            self.Wu = nn.Parameter(torch.empty(size=(3 * emb_dimension, emb_dimension)))
            self.Wv = nn.Parameter(torch.empty(size=(3 * emb_dimension, emb_dimension)))

            nn.init.xavier_uniform_(self.Wuf.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wvf.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wut.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wvt.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wu.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wv.data, gain=1.414)

        else:
            self.Wu1 = nn.Parameter(torch.empty(
                size=(feature_dimension, 10 * emb_dimension)))
            self.Wv1 = nn.Parameter(torch.empty(
                size=(feature_dimension, 10 * emb_dimension)))
            self.Wu2 = nn.Parameter(torch.empty(
                size=(10 * emb_dimension, emb_dimension)))
            self.Wv2 = nn.Parameter(torch.empty(
                size=(10 * emb_dimension, emb_dimension)))

            nn.init.xavier_uniform_(self.Wu1.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wv1.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wu2.data, gain=1.414)
            nn.init.xavier_uniform_(self.Wv2.data, gain=1.414)
        self.act = -1
        if self.act_func == 'relu':
            self.act = nn.ReLU()
        elif self.act_func == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1)
        elif self.act_func == 'prelu':
            self.act = nn.PReLU()
        print('act_func_str:',self.act_func)
        print('act_func: ',self.act)
    def forward(self, pos_u, pos_v, neg_v, v_time, v_his, v_his_times):
        if self.use_fea == 'one-hot':
            emb_u = self.u_embeddings(pos_u)
            emb_v = self.v_embeddings(pos_v)
            emb_neg_v = self.v_embeddings(neg_v)
        elif self.use_fea == 'mix':
            if self.act_func!='none':
                emb_u = self.act(torch.mm(pos_u[0], self.Wu1))  # batch_size*128
                emb_v = self.act(torch.mm(pos_v[0], self.Wv1))
                emb_neg_v = self.act(torch.matmul(neg_v[0], self.Wv1))  # batch_size*neg_num*128
                emb_his = self.act(torch.matmul(v_his[0], self.Wv1))

                emb_u = torch.cat([emb_u, self.u_embeddings(pos_u[1])], dim=1)  # batch_size*256
                emb_v = torch.cat([emb_v, self.v_embeddings(pos_v[1])], dim=1)
                emb_neg_v = torch.cat([emb_neg_v, self.v_embeddings(neg_v[1])], dim=2)  # batch_size*neg_num*256
                emb_his = torch.cat([emb_his, self.v_embeddings(v_his[1])], dim=2)

                emb_u = self.act(torch.mm(emb_u, self.Wu2))  # batch_size*128
                emb_v = self.act(torch.mm(emb_v, self.Wv2))
                emb_neg_v = self.act(torch.matmul(emb_neg_v, self.Wv2))  # batch_size*neg_num*128
                emb_his = self.act(torch.matmul(emb_his, self.Wv2))
            else:
                emb_u = torch.mm(pos_u[0], self.Wu1)  # batch_size*128
                emb_v = torch.mm(pos_v[0], self.Wv1)
                emb_neg_v = torch.matmul(neg_v[0], self.Wv1)  # batch_size*neg_num*128
                emb_his = torch.matmul(v_his[0], self.Wv1)

                emb_u = torch.cat([emb_u, self.u_embeddings(pos_u[1])], dim=1)  # batch_size*256
                emb_v = torch.cat([emb_v, self.v_embeddings(pos_v[1])], dim=1)
                emb_neg_v = torch.cat([emb_neg_v, self.v_embeddings(neg_v[1])], dim=2)  # batch_size*neg_num*256
                emb_his = torch.cat([emb_his, self.v_embeddings(v_his[1])], dim=2)

                emb_u = torch.mm(emb_u, self.Wu2)  # batch_size*128
                emb_v = torch.mm(emb_v, self.Wv2)
                emb_neg_v = torch.matmul(emb_neg_v, self.Wv2)  # batch_size*neg_num*128
                emb_his = torch.matmul(emb_his, self.Wv2)

            # TODO
            att = F.softmax(((emb_u.unsqueeze(1) - emb_his) ** 2).sum(dim=2).neg(), dim=1)

            # att = F.softmax(torch.bmm(emb_his, emb_u.unsqueeze(2)).sum(dim=2).neg(), dim=1)
            # p_mu = ((emb_u - emb_v) ** 2).sum(dim=1).neg()
            p_mu = torch.sum(torch.mul(emb_u, emb_v), dim=1)
            # p_alpha = ((emb_his - emb_v.unsqueeze(1)) ** 2).sum(dim=2).neg()
            p_alpha = torch.bmm(emb_his, emb_v.unsqueeze(2)).sum(dim=2)
            #
            delta = self.delta[pos_u[1]].unsqueeze(1)
            d_time = torch.abs(v_time.unsqueeze(1) - v_his_times)
            p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time))).sum(dim=1)
            # p_lambda = p_mu
            # print('p_lambda:', p_lambda)

            # n_mu = ((emb_u.unsqueeze(1) - emb_neg_v) ** 2).sum(dim=2).neg()
            n_mu = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).sum(dim=2)

            # n_alpha = ((emb_his.unsqueeze(2) - emb_neg_v.unsqueeze(1)) ** 2).sum(dim=3).neg()
            n_alpha = torch.matmul(emb_his.unsqueeze(2).unsqueeze(2), emb_neg_v.unsqueeze(1).unsqueeze(-1)).squeeze()
            n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2))).sum(
                dim=1)
            # n_lambda = n_mu
            # print('n_lambda:', n_lambda)
        elif self.use_fea == 'all':
            emb_u = torch.mm(pos_u[0], self.Wuf)
            emb_v = torch.mm(pos_v[0], self.Wvf)
            emb_neg_v = torch.matmul(neg_v[0], self.Wvf)

            emb_u = torch.cat([emb_u, torch.mm(pos_u[2], self.Wut)], dim=1)
            emb_v = torch.cat([emb_v, torch.mm(pos_v[2], self.Wvt)], dim=1)
            emb_neg_v = torch.cat([emb_neg_v, torch.matmul(neg_v[2], self.Wvt)], dim=2)

            emb_u = torch.cat([emb_u, self.u_embeddings(pos_u[1])], dim=1)
            emb_v = torch.cat([emb_v, self.v_embeddings(pos_v[1])], dim=1)
            emb_neg_v = torch.cat([emb_neg_v, self.v_embeddings(neg_v[1])], dim=2)

            emb_u = torch.mm(emb_u, self.Wu)
            emb_v = torch.mm(emb_v, self.Wv)
            emb_neg_v = torch.matmul(emb_neg_v, self.Wv)
        else:
            emb_u = torch.mm(pos_u, self.Wu1)
            emb_v = torch.mm(pos_v, self.Wv1)
            emb_neg_v = torch.matmul(neg_v, self.Wv1)
            emb_u = torch.mm(emb_u, self.Wu2)
            emb_v = torch.mm(emb_v, self.Wv2)
            emb_neg_v = torch.matmul(emb_neg_v, self.Wv2)

        # score = torch.sum(torch.mul(emb_u, emb_v), dim=1)  # batch_size*1
        # score = ((emb_u - emb_v) ** 2).sum(dim=1).neg()

        score = -torch.log(torch.sigmoid(p_lambda) + 1e-6)
        # print('score:', score)
        # neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze(-1)
        # bmm(batch_size*neg_num*128,batch_size*128*1) -> batch_size*neg_num*1
        # neg_score = ((emb_neg_v - emb_u.unsqueeze(1)) ** 2).sum(dim=2).neg()

        neg_score = - torch.log(
            torch.sigmoid(torch.neg(n_lambda)) + 1e-6).sum(dim=1)
        # print('p_lambda shape:', p_lambda.shape)
        # print('n_lambda shape:', n_lambda.shape)
        # print('neg_socre:', neg_score)
        return torch.mean(score + neg_score)

    def save_embedding(self, feature, id2word, file_name, trans):
        if self.use_fea == 'one-hot':
            embedding = self.u_embeddings.weight.cpu().data
        elif self.use_fea == 'mix':
            w1 = self.Wu1.cpu().data
            w2 = self.Wu2.cpu().data
            if self.act_func=='none':
                embedding = torch.mm(feature, w1)
                embedding = torch.cat(
                    [embedding, self.u_embeddings.weight.cpu().data], dim=1)
                embedding = torch.mm(embedding, w2).numpy()
            else:
                w1 = self.Wu1.data
                w2 = self.Wu2.data
                embedding = self.act(torch.mm(feature.cuda(), w1))
                embedding = torch.cat(
                    [embedding, self.u_embeddings.weight.data], dim=1)
                embedding = self.act(torch.mm(embedding, w2)).cpu().numpy() 
            # embedding = self.node_emb.cpu().data.numpy()
        elif self.use_fea == 'all':
            wf = self.Wuf.cpu().data
            wt = self.Wut.cpu().data
            w = self.Wu.cpu().data
            embedding = torch.mm(feature, wf)
            embedding = torch.cat([embedding, torch.mm(trans, wt)], dim=1)
            embedding = torch.cat([embedding, self.u_embeddings.weight.cpu().data], dim=1)
            embedding = torch.mm(embedding, w).numpy()
        else:
            w1 = self.Wu1.cpu().data
            w2 = self.Wu2.cpu().data
            embedding = torch.mm(feature, w1)
            embedding = torch.mm(embedding, w2).numpy()
        # np.save(file_name, embedding)

        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))

    def eva_embedding(self, word2id, label, n_label, feature, trans):
        if self.use_fea == 'one-hot':
            embedding = self.u_embeddings.weight.cpu().data
        elif self.use_fea == 'mix':
            w1 = self.Wu1.cpu().data
            w2 = self.Wu2.cpu().data
            if self.act_func=='none':
                embedding = torch.mm(feature, w1)
                embedding = torch.cat(
                    [embedding, self.u_embeddings.weight.cpu().data], dim=1)
                embedding = torch.mm(embedding, w2).numpy()
            else:
                w1 = self.Wu1.data
                w2 = self.Wu2.data
                embedding = self.act(torch.mm(feature.cuda(), w1))
                embedding = torch.cat(
                    [embedding, self.u_embeddings.weight.data], dim=1)
                embedding = self.act(torch.mm(embedding, w2)).cpu().numpy()
        elif self.use_fea == 'all':
            wf = self.Wuf.cpu().data
            wt = self.Wut.cpu().data
            w = self.Wu.cpu().data
            embedding = torch.mm(feature, wf)
            embedding = torch.cat([embedding, torch.mm(trans, wt)], dim=1)
            embedding = torch.cat(
                [embedding, self.u_embeddings.weight.cpu().data], dim=1)
            embedding = torch.mm(embedding, w).numpy()
        else:
            w1 = self.Wu1.cpu().data
            w2 = self.Wu2.cpu().data
            embedding = torch.mm(feature, w1)
            embedding = torch.mm(embedding, w2).numpy()
        embedding_dict = {}
        for node in label:
            if (word2id.get(node, -1) == -1):
                print(node)
            embedding_dict[node] = embedding[word2id[node]].tolist()

        NMI = 0
        mi_all = 0
        ma_all = 0
        n = 1
        for i in range(n):
            NMI = NMI + self.evaluate_cluster(embedding_dict, label, n_label)
            micro_f1, macro_f1 = self.evaluate_clf(embedding_dict, label)
            mi_all += micro_f1
            ma_all += macro_f1
        NMI = NMI / n
        micro_f1 = mi_all / n
        macro_f1 = ma_all / n
        print('NMI = %.4f' % NMI)
        print('Micro_F1 = %.4f, Macro_F1 = %.4f' % (micro_f1, macro_f1))

    def evaluate_cluster(self, embedding_dict, label, n_label):
        X = []
        Y = []
        for p in label:
            X.append(embedding_dict[p])
            Y.append(label[p])

        Y_pred = KMeans(n_label, random_state=0).fit(np.array(X)).predict(X)
        nmi = normalized_mutual_info_score(np.array(Y), Y_pred)
        return nmi

    def evaluate_clf(self, embedding_dict, label):
        X = []
        Y = []
        for p in label:
            X.append(embedding_dict[p])
            Y.append(label[p])

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        LR = LogisticRegression()
        LR.fit(X_train, Y_train)
        Y_pred = LR.predict(X_test)

        micro_f1 = f1_score(Y_test, Y_pred, average='micro')
        macro_f1 = f1_score(Y_test, Y_pred, average='macro')
        return micro_f1, macro_f1


class HFSGNSTrainer:

    def __init__(self, args):  # , g_hin):
        # min_cont & care_type
        self.data = DataReader(args.walk_time_file, args.walk_file, args.fea_dim,
                               args.feature_file, args.label_file, args.use_fea, 0, 0, args.trans_file, args.trans_dim,
                               args.dim, args.set_normal)

        dataset = RandomWalkDataset(
            self.data, args.window_size, args.neg_num)

        self.batch_size = args.batch_size
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=args.num_workers, collate_fn=dataset.collate)
        self.output_file_name = args.out_emd_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = args.dim
        self.epochs = args.epochs
        self.label = self.data.word2label
        self.n_label = len(self.data.labels)
        self.initial_lr = args.initial_lr
        self.fea_dim = self.data.feature.shape[1]
        self.neg_k = args.neg_num
        self.use_fea = args.use_fea
        self.trans_dim = args.trans_dim
        self.act_func = args.act_func
        self.skip_gram_model = FeatureSkipGramModel(
            self.fea_dim, self.emb_dimension, args.neg_num, args.use_fea, self.emb_size, args.trans_dim,
            args.set_normal, self.act_func)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.skip_gram_model = nn.DataParallel(self.skip_gram_model)

        if self.use_cuda:
            self.skip_gram_model.to(self.device)

    def train(self):
        print("Training: ", datetime.datetime.now())
        print("Training with act_func:{}".format(self.act_func))
        with torch.no_grad():    
            if isinstance(self.skip_gram_model, torch.nn.DataParallel):
                self.skip_gram_model.module.eva_embedding(
                    self.data.word2id, self.label, self.n_label, self.data.feature, self.data.trans)
            else:
                self.skip_gram_model.eva_embedding(
                    self.data.word2id, self.label, self.n_label, self.data.feature, self.data.trans)

        for epoch in range(self.epochs):
            # TODO 优化器是定义在epoch外还是epoch内?
            optimizer = optim.Adam(
                self.skip_gram_model.parameters(), lr=self.initial_lr)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, len(self.dataloader))
            epoch_loss = 0.0
            n = 0
            start = time.time()
            for bs_num, sample_batched in enumerate(self.dataloader):
                # print("#################:", sample_batched[3])
                # print("#################:", sample_batched[4])
                # print("#################:", sample_batched[5])
                if len(sample_batched[0]) > 1:
                    # for i in range(0, len(sample_batched[0]), self.batch_size*10):
                    if self.use_fea == 'one-hot':
                        pos_u = sample_batched[0].to(self.device)
                        pos_v = sample_batched[1].to(self.device)
                        neg_v = sample_batched[2].to(self.device)
                    elif self.use_fea == 'mix':
                        pos_u = (
                            self.data.feature[sample_batched[0]].to(self.device), sample_batched[0].to(self.device))
                        pos_v = (
                            self.data.feature[sample_batched[1]].to(self.device), sample_batched[1].to(self.device))
                        neg_v = (
                            self.data.feature[sample_batched[2]].to(self.device), sample_batched[2].to(self.device))

                    elif self.use_fea == 'all':
                        pos_u = (
                            self.data.feature[sample_batched[0]].to(self.device), sample_batched[0].to(self.device),
                            self.data.trans[sample_batched[0]].to(self.device))
                        pos_v = (
                            self.data.feature[sample_batched[1]].to(self.device), sample_batched[1].to(self.device),
                            self.data.trans[sample_batched[1]].to(self.device))
                        neg_v = (
                            self.data.feature[sample_batched[2]].to(self.device), sample_batched[2].to(self.device),
                            self.data.trans[sample_batched[2]].to(self.device))

                    else:
                        pos_u = self.data.feature[sample_batched[0]].to(self.device)
                        pos_v = self.data.feature[sample_batched[1]].to(self.device)
                        neg_v = self.data.feature[sample_batched[2]].to(self.device)
                    # .view(pos_u[0].shape[0], self.neg_k, self.fea_dim)

                    v_time = sample_batched[3].to(self.device)
                    # for -u = mix
                    v_his = (self.data.feature[sample_batched[4]].to(self.device), sample_batched[4].to(self.device))
                    # print('line 667: v_his:', v_his)
                    v_his_times = sample_batched[5].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()

                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v, v_time, v_his, v_his_times)
                    # write.add_graph(self.skip_gram_model,
                    #                 (pos_u, pos_v, neg_v))
                    loss = torch.mean(loss)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    n += 1
                    torch.cuda.empty_cache()
                    # print('\rEpoch{}, loss:{}, progress:{}/{} {}%, time:{}s, speed:{} batches/s'.format(
                    #     epoch, round(epoch_loss/n, 2), bs_num, len(self.dataloader), bs_num*100//len(self.dataloader), round(time.time()-start, 2), round(bs_num/(time.time()-start), 2)), end='', flush=True)

            print("\nepoch:" + str(epoch) + " Loss: " +
                  str(round(epoch_loss / n, 2)), datetime.datetime.now())
            with torch.no_grad():      
                if isinstance(self.skip_gram_model, torch.nn.DataParallel):
                    self.skip_gram_model.module.eva_embedding(
                        self.data.word2id, self.label, self.n_label, self.data.feature, self.data.trans)
                    if (epoch + 1) == self.epochs:
                        self.skip_gram_model.module.save_embedding(
                            self.data.feature, self.data.id2word, self.output_file_name, self.data.trans)
                else:
                    self.skip_gram_model.eva_embedding(
                        self.data.word2id, self.label, self.n_label, self.data.feature, self.data.trans)
                    if (epoch % 10 == 0 and epoch!=0):
                        self.skip_gram_model.save_embedding(
                            self.data.feature, self.data.id2word, self.output_file_name + '_epoch_{}'.format(epoch),
                            self.data.trans)
                    if (epoch + 1) == self.epochs:
                        self.skip_gram_model.save_embedding(
                            self.data.feature, self.data.id2word, self.output_file_name, self.data.trans)
