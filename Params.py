import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=256, type=int, help='batch size')
	parser.add_argument('--test_batch', default=2048, type=int, help='batch size in testset')
	parser.add_argument('--reg', default=3e-2, type=float, help='weight decay regularizer')
	parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
	parser.add_argument('--decay', default=0.96, type=float, help='weight decay rate')
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')
	parser.add_argument('--latdim', default=32, type=int, help='embedding size')
	parser.add_argument('--sampNum', default=40, type=int, help='batch size for sampling')
	parser.add_argument('--att_head', default=2, type=int, help='number of attention heads') # 微调
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
	parser.add_argument('--hgnn_layer', default=1, type=int, help='number of hgnn layers')
	parser.add_argument('--hyperNum', default=128, type=int, help='number of hyper edges')
	parser.add_argument('--interestNum', default=2, type=int, help='number of interests')
	parser.add_argument('--trnNum', default=10000, type=int, help='number of training instances per epoch')
	parser.add_argument('--load_model', default=None, help='model name to load')
	parser.add_argument('--shoot', default=20, type=int, help='K of top k')
	parser.add_argument('--data', default='yelp', type=str, help='name of dataset')
	parser.add_argument('--mult', default=100, type=float, help='multiplier for the result')
	parser.add_argument('--keepRate', default=0.5, type=float, help='rate for dropout')
	parser.add_argument('--tstEpoch', default=3, type=int, help='number of epoch to test while training')
	parser.add_argument('--leaky', default=0.5, type=float, help='slope for leaky relu')
	parser.add_argument('--gcn_hops', default=2, type=int, help='number of hops in gcn precessing') # 
	parser.add_argument('--lambda_u', default=1e-4, type=float, help='weight for uuPreLoss')
	parser.add_argument('--ssl_reg', default=1e-4, type=float, help='reg weight for ssl loss')
	parser.add_argument('--ssu_reg', default=1e-4, type=float, help='reg weight for ssu loss')
	parser.add_argument('--edgeSampRate', default=0.1, type=float, help='Ratio of sampled edges')
	return parser.parse_args()
args = parse_args()
args.decay_step = args.trnNum//args.batch