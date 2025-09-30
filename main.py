import argparse
import os
import random
import time
import numpy as np
import torch
from loguru import logger
import ast
from data_setp import DataSet
from model import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args for CNRE model', add_help=False)

    parser.add_argument('--tp', type=str, default='default', help='Task phase or type')
    parser.add_argument('--data_name', type=str, default='taobao', help='Dataset name: beibei or taobao')
    parser.add_argument('--behaviors', help='List of behaviors, set automatically based on data_name', action='append')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on')
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed for reproducibility, matching the paper\'s experiments')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='Testing batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--model_path', type=str, default='./check_point', help='Path to save the model')
    parser.add_argument('--model_name', type=str, default='CNRE', help='Name of the model')

    parser.add_argument('--topk', type=list, default=[10, 20, 50], help='Top-K values for evaluation')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='Evaluation metrics')

    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='L2 regularization weight')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers for the initial intrinsic LightGCN encoder')
    parser.add_argument('--node_dropout', type=float, default=0.1, help='Dropout rate for embeddings')

    parser.add_argument('--layers_nums', type=str, default="[1, 1, 3]",
                        help='Number of GCN layers for each behavior in cascading order (e.g., view, cart, buy)')
    parser.add_argument('--hyper_nums', type=int, default=32, help='Number of hyperedges for the hypergraph module')
    parser.add_argument('--conf_threshold', type=float, default=0.43,
                        help='Purchase confidence threshold for medium preference path')

    args = parser.parse_args()
    args.layers_nums = ast.literal_eval(args.layers_nums)

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if args.data_name == 'beibei':
        args.data_path = f'./data/beibei/'
        args.behaviors = ['view', 'cart', 'buy']
    elif args.data_name == 'taobao':
        args.data_path = f'./data/taobao/'
        args.behaviors = ['view', 'cart', 'buy']
    elif args.data_name == 'tmall':
        args.data_path = f'./data/tmall/'
        args.behaviors = ['view','collect', 'cart', 'buy']
    else:
        raise Exception('data_name cannot be None or is not supported')

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    start_time = time.time()

    dataset = DataSet(args)
    model = Model(args, dataset)

    logger.info("--- Experiment Arguments ---")
    logger.info(args)
    logger.info("--- Model Architecture ---")
    logger.info(model)

    trainer = Trainer(model, dataset, args)

    print("\n--- Starting Training ---")
    trainer.train_model()

    end_time = time.time()
    print(f"Total running time: {end_time - start_time:.2f}s")
