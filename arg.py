import argparse

def dependent_setting(args):
    spt = args.feature_extract.split("_")
    args.feature_backbone = spt[0]
    args.feature_type = spt[1]
        
    if args.imple_flag == "train":
        args.retrieve_triplet = True
        args.iterations = 30000
        args.cycles = 4
    elif args.imple_flag == "extract":
        if args.dataset == "vcdb":
            args.retrieve_triplet = True
            args.iterations = 30000
        else:
            args.retrieve_triplet = False
            args.iterations = -1
        args.cycles = 1

    elif args.imple_flag == "eval":
        args.retrieve_triplet = False
        args.iterations = -1
        args.cycles = 1
    else:
        import pdb; pdb.set_trace()


def arg_func():
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', help='gpu', default=0)

    parser.add_argument('--imple_flag', default=None, required=True, choices=['train', 'extract', 'eval'])

    parser.add_argument('--feats_load_dir', help='feats_load_dir', default=None, type=str)
    parser.add_argument('--save_path', help='save_path', type=str)
    parser.add_argument('--load_path', help='load_path', type=str)

    parser.add_argument('--cycles', help='cycles', default=4)
    parser.add_argument('--batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', help='learning_rate', default=1e-5, type=float)
    parser.add_argument('--weight_decay', help='weight_decay', default=0, type=float)

    parser.add_argument('--dataset', default=None, required=True, choices=['vcdb', 'fivr5k', 'cc_web','fivr200k'], help="dataset")

    parser.add_argument('--feature_extract', default="resnet50_l4imac")
    parser.add_argument('--feature_backbone', default=None, help="If you specified option feature_extract, you don't need to touch this option.")
    parser.add_argument('--feature_type', default=None, help="If you specified option feature_extract, you don't need to touch this option.")

    parser.add_argument('--base_normalize', help='base_normalize', action='store_true')
    parser.add_argument('--pca_whitening', help='whitening', action='store_true')
    parser.add_argument('--pca_reduction', help='reduction', default=None, type=int)

    parser.add_argument('--use_oracle', help='use_oracle', action='store_true')
    parser.add_argument('--fg_gt_label', help='fg_gt_label', default=None, type=str, choices=["n", "ns", "nsh"])

    parser.add_argument('--retrieve_triplet', help='retrieve_triplet', action='store_true')
    parser.add_argument('--extract_fixed', help='extract_fixed', default=None, type=str)
    parser.add_argument('--suppression', help='suppression', default="vvs", type=str, choices=["vvs"])

    # for ablation argument
    parser.add_argument('--vvs_vinit', help='vvs_vinit', default="v_t", type=str, choices=["v_t", "const", "rand"])
    parser.add_argument('--vvs_h_connect', help='vvs_h_connect', action='store_true')

    parser.add_argument('--vvs_sigmoid_T', help='vvs_sigmoid_T', default=512.0, type=float)
    parser.add_argument('--vvs_sigmoid_T_tsm', help='ablation_for_tempered_sigmoid', default=1.0, type=float)
    parser.add_argument('--vvs_sigmoid_T_ddm', help='ablation_for_tempered_sigmoid', default=1.0, type=float)


    parser.add_argument('--refinement', help='refinement', action='store_true')
    parser.add_argument('--vvs_tsm', help='vvs_tsm', action='store_true')
    parser.add_argument('--vvs_tgm', help='vvs_tgm', action='store_true')
    parser.add_argument('--vvs_ddm', help='vvs_ddm', action='store_true')
    
    parser.add_argument('--weight_triplet', help='triplet_loss_ratio', default=1.0, type=float)
    parser.add_argument('--weight_saliency', help='saliency_loss_ratio', default=1.0, type=float)
    parser.add_argument('--weight_frame', help='frame_loss_ratio', default=1.0, type=float)
    parser.add_argument('--weight_ddm', help='frame_loss_ratio', default=0.5, type=float)

    parser.add_argument('--thresh_s', help='thresh_s', default=0.5, type=float)
    
    parser.add_argument('--distractor_sampling_ratio', help='distractor_sampling_ratio', default="20_50", type=str, choices=["0_20",'20_50','50_80','80_100'])
    
    parser.add_argument('--long_term_retrieval', help='long_term_retrieval', action='store_true')
    parser.add_argument('--average_eval', help='average_eval', action='store_true')
    parser.add_argument('--neg_len', help='neg_len', default=32, type=int)
    parser.add_argument('--mag_opt', help='mag_opt', default=5, type=int, choices=[1,2,3,4,5,6,7,8,9])
    parser.add_argument('--mag_thresh', help='mag_thresh', default=40, type=int)
    parser.add_argument('--baseline', help='baseline', action='store_true')

    args = parser.parse_args()

    dependent_setting(args)

    vars_args = vars(args)
    max_klen = max([len(i) for i in list(vars_args.keys())])
    max_vlen = max([len(str(i)) for i in list(vars_args.values()) if i is not None])
    print("\n\n")
    header = int((max_klen+max_vlen-8)/2)*"*" + " Args List " + int((max_klen+max_vlen-8)/2)*"*"
    print("\t"+header)
    for arg in vars_args:
        left = (max_klen-len(arg)) * " " + arg
        print("\t"+"{} : {}".format(left, getattr(args, arg)))
    print("\t"+"*" * len(header))
    print("\n\n")

    return args