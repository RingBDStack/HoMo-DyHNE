import argparse
from SGNS import *
# from FSGNS import *
# from FASGNS import *
# from HSGNS import *
# from DFASGNS import *
from HFSGNS import *
# from IHFSGNS import *
# from DGIFSGNS import *
# from FSGNS_infonce import *

import warnings

warnings.filterwarnings('ignore')


def main():
    args = init_para()
    print('epoches:',args.epochs)
    print('walk_file:{}\nwindow_size:{}\nlearning_rate:{}'.format(args.walk_file, args.window_size, args.initial_lr))
    print('out_file:',args.out_emd_file)
    if args.model == 'SGNS':
        print('use model SGNS!')
        model = SGNSTrainer(args)  # , g_hin)
    # elif args.model == 'FSGNS':
    #     model = FSGNSTrainer(args)  # , g_hin)
    # elif args.model == 'FASGNS':
    #     model = FASGNSTrainer(args)  # , g_hin)
    # elif args.model == 'DFASGNS':
    #     model = DFASGNSTrainer(args)
    # elif args.model == 'HSGNS':
    #     print('use model HSGNS!')
    #     model = HSGNSTrainer(args)
    elif args.model == 'HFSGNS':
        print('use model HFSGNS!')
        model = HFSGNSTrainer(args)
    # elif args.model == 'IHFSGNS':
    #     print('use model IHFSGNS!')
    #     model = IHFSGNSTrainer(args)
    # elif args.model == 'DGIFSGNS':
    #     print('use model DGIFSGNS!')
    #     model = DGIFSGNSTrainer(args)
    # elif args.model == 'FSGNSInce':
    #     print('use model FSGNSInce!')
    #     model = FSGNSInceTrainer(args)
    model.train()


def init_para():
    parser = argparse.ArgumentParser(description="FSG")
    parser.add_argument('-d', '--dataset', default='acm', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='SGNS', type=str, help='Train model')
    parser.add_argument('-w', '--window_size', default=5,
                        type=int, help='Window size')
    parser.add_argument('-k', '--neg_num', default=5,
                        type=int, help='Negative num')
    parser.add_argument('-bs', '--batch_size', default=32,
                        type=int, help='Batch size')
    parser.add_argument('-fd', '--fea_dim', default=196, type=int, help='Feature dimension')
    parser.add_argument('-td', '--trans_dim', default=3, type=int, help='Trans dimension')
    parser.add_argument('-ed', '--dim', default=128, type=int, help='Embedding dimension')
    parser.add_argument('-lr', '--initial_lr', default=0.025,
                        type=float, help='Learning rate')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='Epochs')
    parser.add_argument('-n', '--num_workers', default=0,
                        type=int, help='Workers')
    parser.add_argument('-u', '--use_fea',
                        default='all', type=str, help='Feature type')
    parser.add_argument('-a', '--alpha',
                        default=0.2, type=float, help='Alpha for leakyrelu')
    parser.add_argument('-dp', '--dropout',
                        default=0.6, type=float, help='Dropout')
    parser.add_argument('-head', '--nheads',
                        default=8, type=int, help='Attention head number')
    parser.add_argument('-ff', '--feature_file',
                        default='datasets/acm/acm.features', type=str, help='Feature file name')
    parser.add_argument('-tf', '--trans_file',
                        default='acm.time.100.1.0.8.trans', type=str, help='Trans file name')
    parser.add_argument('-lf', '--label_file',
                        default='datasets/acm/acm.labels', type=str, help='Label file name')
    parser.add_argument('-of', '--out_emd_file',
                        default='acm.SGNS.embedding.txt', type=str, help='Out file name')
    parser.add_argument('-wf', '--walk_file', default='acm.time.100.1.0.8.walk', type=str, help='Walk file name')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('-ef', '--edge_file', default='datasets/acm/acm.edges', type=str, help='edge file name')

    parser.add_argument('-wtf', '--walk_time_file', default='Aminer.time.40.80.0.8.times', type=str,
                        help='time file name')
    parser.add_argument('-act', '--act_func', default='none', type=str)
    parser.add_argument('-wl', '--walk_length', default=10, type=int)
    parser.add_argument('-nom', '--set_normal', default=0, type=int, 
                        help='1 for normalization of feature 0 for not, for Aminer and Dblp, set nom = 1, for taobao , set nom = 0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
    print('done!')
