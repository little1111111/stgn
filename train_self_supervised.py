import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from tool.utils import EarlyStopMonitor, get_neighbor_finder
from tool.data_processing import get_data_node_classification, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='cert')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                   'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=11, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--use_validation', action='store_true',
                    help='Whether to use a validation set')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
full_data, node_features, edge_features, train_data, val_data, test_data = \
    get_data_node_classification(DATA, use_validation=args.use_validation)


# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)


class NegativeSampler:
    def __init__(self, train_data, full_data, seed=42):
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

        # 获取训练集中的所有节点
        self.train_nodes = set(train_data.sources) | set(train_data.destinations)

        # 获取负样本节点（在训练集中的）
        negative_full_mask = full_data.labels == 1
        neg_src = set(full_data.sources[negative_full_mask])
        neg_dst= set(full_data.destinations[negative_full_mask])
        # 只保留在训练集中的负样本节点
        self.valid_neg_nodes = list(neg_src & self.train_nodes | neg_dst & self.train_nodes)

        if len(self.valid_neg_nodes) == 0:
            logger.warning("No valid negative samples found in training set!")
            # 如果没有负样本，使用训练集中的所有节点作为备选
            self.valid_neg_nodes = list(self.train_nodes)

        logger.info(f"Number of valid negative samples: {len(self.valid_neg_nodes)}")

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        """采样指定数量的负样本源节点"""
        if len(self.valid_neg_nodes) < size:
            # 如果负样本数量不足，进行重复采样
            return self.random_state.choice(self.valid_neg_nodes, size=size, replace=True)
        else:
            return self.random_state.choice(self.valid_neg_nodes, size=size, replace=False)
            
    def sample_pair(self, size):
        """采样指定数量的负样本源节点-负样本目的节点对"""
        # 采样负样本源节点
        neg_source_nodes = self.sample(size)
        # 采样负样本目的节点
        neg_destination_nodes = self.sample(size)
        
        # 确保负样本源节点和负样本目的节点不相等
        for i in range(size):
            while neg_source_nodes[i] == neg_destination_nodes[i]:
                neg_destination_nodes[i] = self.sample(1)[0]
                
        return neg_source_nodes, neg_destination_nodes


# 修改负样本采样器的初始化
negative_edge_sampler = NegativeSampler(train_data, full_data,seed=42)


for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=edge_features, device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    tgn = tgn.to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    val_aps = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()
        ### Training

        # Reinitialize memory of the model at the start of each epoch
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        for k in range(0, num_batch, args.backprop_every):
            loss = 0
            optimizer.zero_grad()

            # Custom loop to allow to perform backpropagation only every a certain number of batches
            for j in range(args.backprop_every):
                batch_idx = k + j

                if batch_idx >= num_batch:
                    continue

                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(num_instance, start_idx + BATCH_SIZE)
                sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                    train_data.destinations[start_idx:end_idx]
                edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
                timestamps_batch = train_data.timestamps[start_idx:end_idx]

                size = len(sources_batch)
                neg_source_nodes, neg_destination_nodes = negative_edge_sampler.sample_pair(size)

                with torch.no_grad():
                    pos_label = torch.ones(size, dtype=torch.float, device=device)
                    neg_label = torch.zeros(size, dtype=torch.float, device=device)

                tgn = tgn.train()
                pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, neg_source_nodes, neg_destination_nodes,
                                                                    timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

                loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

            loss /= args.backprop_every

            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())

            # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
            # the start of time
            if USE_MEMORY:
                tgn.memory.detach_memory()

        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        # Validation uses the full graph
        tgn.set_neighbor_finder(full_ngh_finder)

        if USE_MEMORY:
            # Backup memory at the end of training, so later we can restore it and use it for the
            # validation on unseen nodes
            train_memory_backup = tgn.memory.backup_memory()

        val_metrics = eval_edge_prediction(model=tgn,
                                         negative_edge_sampler=negative_edge_sampler,
                                         data=val_data,
                                         n_neighbors=NUM_NEIGHBORS)
        if USE_MEMORY:
            val_memory_backup = tgn.memory.backup_memory()
            # Restore memory we had at the end of training to be used when validating on new nodes.
            # Also backup memory after validation so it can be used for testing (since test edges are
            # strictly later in time than validation edges)
            tgn.memory.restore_memory(train_memory_backup)

        if USE_MEMORY:
            # Restore memory we had at the end of validation
            tgn.memory.restore_memory(val_memory_backup)

        val_aps.append(val_metrics["mean_ap"])
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aps": val_aps,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('Validation metrics:')
        logger.info(f'AP: {val_metrics["mean_ap"]:.4f}')
        logger.info(f'AUC: {val_metrics["mean_auc"]:.4f}')
        logger.info(f'Precision: {val_metrics["precision"]:.4f}')
        logger.info(f'Recall: {val_metrics["recall"]:.4f}')
        logger.info(f'F1: {val_metrics["f1"]:.4f}')

        # Early stopping
        if early_stopper.early_stop_check(val_metrics["mean_ap"]):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgn.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgn.eval()
            break
        else:
            torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

    # Training has finished, we have loaded the best model, and we want to backup its current
    # memory (which has seen validation edges) so that it can also be used when testing on unseen
    # nodes
    if USE_MEMORY:
        val_memory_backup = tgn.memory.backup_memory()

    ### Test
    tgn.embedding_module.neighbor_finder = full_ngh_finder
    test_metrics = eval_edge_prediction(model=tgn,
                                      negative_edge_sampler=negative_edge_sampler,
                                      data=test_data,
                                      n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
        tgn.memory.restore_memory(val_memory_backup)

    logger.info('Test metrics:')
    logger.info(f'AP: {test_metrics["mean_ap"]:.4f}')
    logger.info(f'AUC: {test_metrics["mean_auc"]:.4f}')
    logger.info(f'Precision: {test_metrics["precision"]:.4f}')
    logger.info(f'Recall: {test_metrics["recall"]:.4f}')
    logger.info(f'F1: {test_metrics["f1"]:.4f}')

    # Save results for this run
    pickle.dump({
        "val_aps": val_aps,
        "test_ap": test_metrics["mean_ap"],
        "epoch_times": epoch_times,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    logger.info('Saving TGN model')
    if USE_MEMORY:
        # Restore memory at the end of validation (save a model which is ready for testing)
        tgn.memory.restore_memory(val_memory_backup)
    torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGN model saved')
