import faiss, argparse, os, sys
import numpy as np


def calc_mAP(q):
	assert type(q) == np.ndarray
	assert q.ndim == 2
	invvalues = np.divide(np.ones(q.shape[1]), np.ones(q.shape[1]).cumsum())

	map_ = 0
	prec_sum = q.cumsum(1)
	for i in range(prec_sum.shape[0]):
		idx_nonzero = np.nonzero(q[i])[0]
		if len(idx_nonzero) > 0:
			map_ += np.mean(prec_sum[i][idx_nonzero] * invvalues[idx_nonzero])

	return map_ / q.shape[0]


def search_topk(X_train, X_base, y_base, X_query, y_query, topk):
	index = faiss.IndexLSH(origin_dim, args.nbits)
	index.train(X_train.astype('float32'))
	index.add(X_base.astype('float32'))
	D, I = index.search(X_query.astype('float32'), int(topk))

	predictions = y_base[I]
	results = np.sum(np.equal(predictions, np.expand_dims(y_query, axis=1)), axis=2)

	return calc_mAP(results)


if __name__=="__main__":
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--code', choices=['lsh', 'onehot', 'none'], default='none')
	parser.add_argument('--labels', type=int, default=1000)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--nbits', type=int, default=48)
	parser.add_argument('--path', type=str, default="./")
	parser.add_argument('--topk', type=int, default=5000)
	args = parser.parse_args()
	#
	coco_query_feat = np.load(os.path.join(args.path, "coco_query_feat.npy"))
	coco_query_labels = np.load(os.path.join(args.path, "coco_query_labels.npy"))

	coco_base_feat = np.load(os.path.join(args.path, "coco_base_feat.npy"))
	coco_base_labels = np.load(os.path.join(args.path, "coco_base_labels.npy"))

	coco_train_feat = np.load(os.path.join(args.path, "coco_train_feat.npy"))
	coco_train_labels = np.load(os.path.join(args.path, "coco_train_labels.npy"))
	nb = coco_base_feat.shape[0]

	origin_dim = 80

	topk = args.topk
	if args.topk>nb:
		topk = nb

	mAP = search_topk(
		coco_train_feat,
		coco_base_feat,
		coco_base_labels,
		coco_query_feat,
		coco_query_labels,
		topk
	)
	print("mAP: {0}".format(mAP))
