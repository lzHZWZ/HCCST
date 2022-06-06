import argparse
from config import get_config

swin_basedir = './PretrainedModels/'
swin_pretrained_pools = {
	'b384_22k':'swin_base_patch4_window12_384_22k.pth',
	'b224_1k':'swin_base_patch4_window7_224.pth',
	'l384_22k':'swin_large_patch4_window12_384_22k.pth',
	't224_1k':'swin_tiny_patch4_window7_224.pth',
}

def par_option():
    parser = argparse.ArgumentParser(description='CHV')
    
    parser.add_argument('--model_type', type=str, default='swin', help='The type of base model')
    parser.add_argument('--resume_path', type=str, default='', help='model resume path')
    parser.add_argument('--resume', action='store_true', help="resume from pretrained model?")
    parser.add_argument('--resume_weight_path',default='',type=str,help='resume weight path')
    parser.add_argument('--data_name', type=str, default='voc', help='voc or coco...')
    parser.add_argument('--data_path', type=str, default='../../data/voc', help='dataset path...')
    
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_center', type=float, default=0.0001, help='learning rate for center update')
    
    parser.add_argument('--epochs', type=int, default=90, help='training epoch')
    parser.add_argument('--use_gpu', type=bool, default=True, help="use gpu ?")
    parser.add_argument('--gpus', type=str, default="0", help="define gpu id")
    
    parser.add_argument('--num_iteration', type=int, default=60, help='number of iteration')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='the batch size for training')  
    
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for SGD')
    parser.add_argument('--eval_frequency', type=int, default=4, help='the evaluate frequency for testing')
    parser.add_argument('--image_size', type=int, default=224, help='image size')
    
    parser.add_argument('--word2vec_file', type=str, default='../data/voc/voc2007_bert768_word2vec.pkl',
                        help='word to vector file path')
    
    parser.add_argument('--num_class', type=int, default=20, help='The number of classes')
    parser.add_argument('--workers', type=int, default=4, help='number of data loader workers.')
    parser.add_argument('--multi_lr', type=float, default=0.01, help='multiplier for learning rate')
    parser.add_argument('--lambda_Q', type=float, default=0.05, help='hyper-parameters for quantization loss')
    parser.add_argument('--lambda1', type=float, default=0.2, help='hyper-parameters 1')
    parser.add_argument('--lambda2', type=float, default=0.05, help='hyper-parameters 1')
    parser.add_argument('--centerLoss_type', type=str, default='BCELoss', help='BCELoss CauchyLoss MarginLoss...')
    parser.add_argument('--eta', type=float, default=0.1,
                        help="hyper-parameters for alternative optimization in solution of center weight updating")
    
    parser.add_argument('--lambda_R', type=float, default=0.001,
                        help="hyper-parameters of alternative optimization in weight regulation")
    
    parser.add_argument('--fixed_weight', action='store_true', help="fix center weight")
    
    parser.add_argument('--centerWeight_path', type=str, default='../data/')
    
    parser.add_argument('--centers_path', type=str, default='../data/voc/16_voc_20_class.pkl')
    
    parser.add_argument('--center_update', action='store_true', help="update hash center or not?")
    parser.add_argument('--w2v_dim', type=int, default=768, help="output dim of word embedding")
    
    parser.add_argument('--hash_bit', type=int, default='64', help='hash bit,it can be 8, 16, 32, 64, 128...')
    parser.add_argument('--batch_size_hash', type=int, default=40,
                        help='the batch size for training') 
    
    parser.add_argument('--radius', type=float, default=2.0, help=" hamming ball radius--MarginLoss")
    parser.add_argument('--gamma', type=float, default=10.0, help="parameter in Cauchy")
    parser.add_argument('--beta', type=float, default=5.0, help='beta in smooth maximum--MarginLoss')
    
    parser.add_argument('--alpha_0', type=float, default=1., help="hyper-parameter for centerLoss in hash center loss ")
    parser.add_argument('--alpha_1', type=float, default=1.0, help="hyper-parameter for inter_loss in hash center loss ")
    parser.add_argument('--alpha_2', type=float, default=0.5, help='hyper-parameter for KL_divergence in hash center loss')
    
    parser.add_argument('--beta_sigmoid', type=float, default=0.1, help="coefficient beta of sigmoid function in BCELoss")

    parser.add_argument('--R', type=int, default=3000, help='MAP@R')
    parser.add_argument('--T', type=float, default=0, help='Threshold for binary')
    parser.add_argument('--model_name', type=str, default='imagenet_64bit_0.873_tiny.pkl',
                        help='Put any model you want to test here')
    parser.add_argument('--start_test_epoch', type=int, default=20, help="start test epoch")

    parser.add_argument('--cfg', type=str, default='./configs/swin_tiny_patch4_window7_224.yaml', metavar="FILE", help='path to config file', )
    parser.add_argument('--load_swin', default=swin_basedir + swin_pretrained_pools['t224_1k'],
                        type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--use_swin_pretrain', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--center_model', default='dual', type=str, help='type of center model')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    
    return args, config
