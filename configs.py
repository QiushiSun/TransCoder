import random
import torch
import logging
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str,default='summarize',
                        choices=['summarize', 'translate', 'refine', 'generate', 'defect', 'clone'])# without complete
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--add_lang_ids", action='store_true')
    # plbart unfinished
    parser.add_argument("--model_name", default="codet5",
                        type=str, choices=['plbart', 'codet5'])
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")  # previous one 42
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--huggingface_locals', type=str, default='data/huggingface_locals',
                    help="directory to save huggingface models")
    parser.add_argument("--cache_path", type=str, default='cache_data')
    parser.add_argument("--res_dir", type=str, default='results',
                        help='directory to save fine-tuning results')
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--model_dir", type=str, default='saved_models',
                        help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard',
                        help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1,
                        help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0,
                        help='index of the gpu to use in a cluster')
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='outputs', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--add_task_prefix", action='store_true',
                        help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true',
                        help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--dev_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for validating.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for testing.")
    parser.add_argument("--attention_batch_size", default=100, type=int,
                        help="Batch size per GPU/CPU for computing attention.")
    parser.add_argument("--is_clone_sample", default=0, type=int,
                        help="clone&defect data is large, 0 for not sample and 1 for sample")                    
    # parser.add_argument('--layer_num', type=int, default=-1,
    #                 help="layer which attention is concerned, -1 for last layer, else for all 0-11 layers")
    # parser.add_argument('--quantile_threshold', type=float, default=0.75,
    #                 help="threshold of quantile which we concern attention should be gt and distance should be lt")
    # parser.add_argument('--frequent_type',  default=1, type=int, choices=[0,1],
    #                 help="whether only use frequent_type")
    # parser.add_argument('--upgraded_ast',  default=1, type=int, choices=[0,1],
    #                 help="whether to use upgraded ast")
    parser.add_argument('--few_shot',  default=64, type=int,
                    help="use k shot, -1 for full data")#####rename to Sample

    parser.add_argument("--prefix_tuning", default=False, type=str,
                    help="parameter-efficient prefix tuning, pass_tuning refers to GAT prefix,\
                    GCN refers to GCN prefix,prefix_tuning refers to MLP prefix",\
                        choices=['pass_tuning','GCN' ,'prefix_tuning', 'False'])
    parser.add_argument("--adapter_tuning", default=0, type=int,
                    help="parameter-efficient adapter tuning, 0 for not tuning, 1 for tuning")#only support codet5 currently
    parser.add_argument("--bitfit", default=0, type=int,
                    help="parameter-efficient bitfit, 0 for not tuning, 1 for tuning")
    
    parser.add_argument("--work_dir", type=str, default='TransCoder',
                        help='work dir')
    parser.add_argument("--prefix_token_level", default='token', type=str,
                        help="how to parse initial prefix code, choose 'token' or 'subtoken' level of ids/init_dist_weight")
    parser.add_argument("--gat_token_num", default=32, type=int,
                        help="number of tokens to use for gat, must be divided with max_source_length in encoder2decoder with no remainder")
    parser.add_argument("--fix_model_param", default=0, type=int,
                    help="when prefix_tuning, fix model param or not ")
    
    parser.add_argument("--knowledge_usage", default='separate', type=str,
                        help="for t5&bart, how knowledge prefix use: separate or concatenate")
    parser.add_argument("--use_description", default=0, type=int,
                    help="use_description or not ")
    parser.add_argument("--concatenate_description", default=0, type=int,
                    help="concatenate_description or not ")
    parser.add_argument("--map_description", default=0, type=int,
                    help="map_description or not ")
    parser.add_argument("--prefix_dropout", default=0.0, type=float,
                        help="prefix_dropout.")


    parser.add_argument("--meta_task", default='translate2cls', type=str,
            help="do meta task",
            choices=['cls2translate','translate2cls','cls2summarize','summarize2cls','translate2summarize','summarize2translate',
            'cross2java','cross2php','cross2ruby','cross2python','cross2go','cross2javascript'])
    parser.add_argument("--shared_dir", default='save_models/shared', type=str,
                    help="directory to store shared state dict list")
    parser.add_argument("--origin_model_dir", default='data/huggingface_locals', type=str,
                    help="directory to store origin model")
    parser.add_argument("--meta_epochs", default=5, type=int,
                    help="meta epochs for source task mata train")
    parser.add_argument("--test_sample_rate", default=1, type=float,
                    help="construct test_sample_rate of test sample as target task to simulate low resource scene, 0.5 for 50% of target task, 1 for full task")
    parser.add_argument("--prefix_type", default='tuned', type=str,
                    help="how to prefix for test, tuned for meta_trained",choices=['tuned','random','False'])
    parser.add_argument("--do_meta_train", default=1, type=int,
                    help="whether do meta train")
    parser.add_argument("--do_meta_test", default=1, type=int,
                    help="whether do meta test")
    parser.add_argument("--debug", default=0, type=int,
                    help="for debug")

    args = parser.parse_args()
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_count = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_count)
    args.device = device
    args.cpu_count = cpu_count


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_hyperparas(args):
    if args.qiangtamadeka:
        args.few_shot = -1

    args.adam_epsilon = 1e-8
    args.beam_size = 10
    args.gradient_accumulation_steps = 1
    args.weight_decay = 0.0
    args.max_source_length = 512
    if args.model_name in ['t5', 'codet5']:
        lr=2e-5
    elif args.model_name in ['bart', 'plbart','unixcoder']:
        lr=2e-5# 5e-5
    elif args.model_name in ['graphcodebert']:
        lr=2e-5# 5e-5(in repo)#1e-4(in plbartpaper)
    if args.task == 'summarize':
            args.data_num = args.few_shot if args.few_shot > 0 else -1
            args.lr = lr if not args.prefix_tuning else 1e-4#2e-3
            args.max_source_length = 256
            args.max_target_length = 128
    elif args.task == 'translate':
        args.data_num = args.few_shot if args.few_shot > 0 else -1
        if args.model_name in ['t5', 'codet5'] and args.sub_task == 'java-cs':
            args.lr = lr if not args.prefix_tuning else 5e-4#0224#2e-3
        else:
            args.lr = lr if not args.prefix_tuning else 1e-4#0224#2e-3
        args.max_source_length = 320
        args.max_target_length = 256
    elif args.task == 'refine':
        args.data_num = args.few_shot if args.few_shot > 0 else -1
        args.lr = lr if not args.prefix_tuning else 1e-4#0224#2e-3
        if args.sub_task == 'small':
            args.max_source_length = 130
            args.max_target_length = 120
        else:
            args.max_source_length = 240
            args.max_target_length = 240
    elif args.task == 'generate':
        args.data_num = args.few_shot if args.few_shot > 0 else -1
        args.lr = lr if not args.prefix_tuning else 5e-4#0224#2e-3
        args.max_source_length = 320
        args.max_target_length = 150
    elif args.task == 'complete':
        args.data_num = args.few_shot if args.few_shot > 0 else -1
        args.lr = 1e-5 if not args.prefix_tuning else 1e-4#1e-3
        args.max_source_length = 256
        args.max_target_length = 256
    elif args.task == 'defect':
        args.data_num = args.few_shot * 2 if args.few_shot > 0 else -1
        args.lr = 8e-6 if not args.prefix_tuning else 5e-4#0224 #8e-6#8e-4
        args.max_source_length = 512
        args.max_target_length = 3  # as do not need to add lang ids
    elif args.task == 'clone':
        args.data_num = args.few_shot * 2 if args.few_shot > 0 else -1 
        args.lr = lr if not args.prefix_tuning else 1e-4
        args.max_source_length = 512#512#400
        args.max_target_length = 512#512#400

    if args.few_shot == -1 or args.few_shot>=2048:
        if args.task in ['clone']:
            args.num_train_epochs = 2 if not args.prefix_tuning else 1# if not torch.cuda.is_available() else 2*torch.cuda.device_count()//2
            #for clone BCB full data!!!
            if args.is_clone_sample:
                args.num_train_epochs = args.num_train_epochs * 10
            args.patience = args.num_train_epochs*1000#min( 10, args.num_train_epochs//5*5)
        elif args.task in ['defect']:
            args.num_train_epochs = 120 if not args.prefix_tuning else 120 #old40 #if not torch.cuda.is_available() else 10*torch.cuda.device_count()//2*2
            # if args.is_clone_sample:
            #     args.num_train_epochs = args.num_train_epochs * 10
            args.patience = args.num_train_epochs*1000#min( 10, args.num_train_epochs//5*5)
        elif args.task in ['generate','translate','summarize']:
            args.num_train_epochs = 30 if not torch.cuda.is_available() else 50*torch.cuda.device_count()#60
            args.patience = min( 10, args.num_train_epochs//5*2)
        else:#refine
            args.num_train_epochs = 30 if not torch.cuda.is_available() else 30*torch.cuda.device_count()#60
            args.patience = min( 10, args.num_train_epochs//5*2)
        # else:
        #     args.num_train_epochs = 60 if not torch.cuda.is_available() else 60*torch.cuda.device_count()//2
        #     args.patience = min( 10, args.num_train_epochs//5*2)
        if args.model_name in ['t5', 'codet5']:
            args.batch_size = 8  if not torch.cuda.is_available() else 8 * torch.cuda.device_count()
        elif args.model_name in ['bart', 'plbart']:
            args.batch_size = 16 if not torch.cuda.is_available() else 16 * torch.cuda.device_count()
        else:
            args.batch_size = 16 if not torch.cuda.is_available() else 16 * torch.cuda.device_count()
        
        # args.batch_size = 2#####################################################
        if args.qiangtamadeka:
            args.batch_size = 16
            args.num_train_epochs = 10000

        # args.batch_size = 128 if args.model_name not in ['t5', 'codet5'] else 16
        args.warmup_steps = 1000
        args.dev_batch_size = args.batch_size * 1 if not torch.cuda.is_available() else args.batch_size//torch.cuda.device_count()*1
        args.test_batch_size = args.batch_size * 1 if not torch.cuda.is_available() else args.batch_size//torch.cuda.device_count()*1
        if args.task in ['refine','generate'] and args.model_name in ['bart', 'plbart']:#3090
            if args.prefix_tuning:
                args.dev_batch_size = args.dev_batch_size // 2 #4
                args.test_batch_size = args.test_batch_size // 2
            else:
                args.dev_batch_size = args.dev_batch_size // 2
                args.test_batch_size = args.test_batch_size // 2

    elif args.few_shot < 128: #16,32,64
        args.num_train_epochs = 64
        # args.lr =5e-8
        args.batch_size = 2
        args.dev_batch_size = args.batch_size
        args.test_batch_size = args.batch_size
    elif args.few_shot < 512: #128,256
        args.num_train_epochs = 48
        args.batch_size = 4 if args.model_name not in ['t5', 'codet5'] else 4
        args.dev_batch_size = args.batch_size
        args.test_batch_size = args.batch_size
    elif args.few_shot < 2048: #512,1024
        args.num_train_epochs = 32
        args.batch_size = 8 if args.model_name not in ['t5', 'codet5'] else 4
        args.dev_batch_size = args.batch_size
        args.test_batch_size = args.batch_size