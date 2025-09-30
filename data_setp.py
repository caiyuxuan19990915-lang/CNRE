import os
import random
import json
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TestDataset(Dataset):
    def __init__(self, user_samples):
        self.user_samples = user_samples

    def __getitem__(self, idx):
        return int(self.user_samples[idx])

    def __len__(self):
        return len(self.user_samples)


class MultiBehaviorTrainDataset(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None, behavior_history=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behaviors = behaviors
        self.behavior_dict = behavior_dict
        self.behavior_history = behavior_history
        self.users = list(self.behavior_dict['all'].keys())
        self.n_behaviors = len(self.behaviors)

    def __getitem__(self, idx):
        user_id_str = self.users[idx]
        user_id_int = int(user_id_str)

        user_all_interacted_items = self.behavior_dict['all'].get(user_id_str, [])
        pos_items_list, neg_items_list, histories_list = [], [], []

        for behavior in self.behaviors:
            user_behavior_items = self.behavior_dict[behavior].get(user_id_str, [])

            if not user_behavior_items:
                pos_item, neg_item = 0, 0
            else:
                pos_item = random.choice(user_behavior_items)
                while True:
                    neg_item = random.randint(1, self.item_count)
                    if neg_item not in user_all_interacted_items:
                        break

            history_vector = self.behavior_history.get((user_id_int, pos_item), [0] * self.n_behaviors)
            pos_items_list.append(pos_item)
            neg_items_list.append(neg_item)
            histories_list.append(history_vector)

        return (
            torch.tensor(user_id_int, dtype=torch.long),
            torch.tensor(pos_items_list, dtype=torch.long),
            torch.tensor(neg_items_list, dtype=torch.long),
            torch.tensor(histories_list, dtype=torch.float)
        )

    def __len__(self):
        return len(self.users)


class DataSet(object):
    def __init__(self, args):
        self.args = args
        self.behaviors = args.behaviors
        self.path = args.data_path

        self.__get_count()
        self.__get_behavior_items()
        self.__get_validation_dict()
        self.__get_test_dict()
        self.__get_sparse_interact_dict()
        self.__get_behavior_history()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_interacts.items() if len(x) > 0])
        self.test_gt_length = np.array([len(x) for _, x in self.test_interacts.items() if len(x) > 0])

    def __get_count(self):
        with open(os.path.join(self.path, 'count.txt'), 'r', encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']

    def __get_behavior_items(self):
        self.train_behavior_dict = {}
        for behavior in self.behaviors:
            with open(os.path.join(self.path, behavior + '_dict.txt'), 'r', encoding='utf-8') as f:
                self.train_behavior_dict[behavior] = json.load(f)
        with open(os.path.join(self.path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
            self.train_behavior_dict['all'] = json.load(f)

    def __get_test_dict(self):
        with open(os.path.join(self.path, 'test_dict.txt'), 'r', encoding='utf-8') as f:
            self.test_interacts = json.load(f)
        self.test_users = list(self.test_interacts.keys())

    def __get_validation_dict(self):
        with open(os.path.join(self.path, 'validation_dict.txt'), 'r', encoding='utf-8') as f:
            self.validation_interacts = json.load(f)
        self.val_users = list(self.validation_interacts.keys())

    def __get_behavior_history(self):
        self.behavior_history = defaultdict(lambda: [0] * len(self.behaviors))
        print("Building behavior history...")
        for i, behavior in enumerate(self.behaviors):
            with open(os.path.join(self.path, behavior + '.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        u, item = int(parts[0]), int(parts[1])
                        self.behavior_history[(u, item)][i] = 1
        print("Behavior history built.")

    def __get_sparse_interact_dict(self):
        self.edge_index = {}
        all_rows, all_cols = [], []
        for behavior in self.behaviors:
            rows, cols = [], []
            with open(os.path.join(self.path, behavior + '.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().split()
                    rows.append(int(line[0]))
                    cols.append(int(line[1]))

            u_nodes = torch.LongTensor(rows)
            i_nodes = torch.LongTensor(cols) + self.user_count + 1

            edge_index = torch.stack([torch.cat([u_nodes, i_nodes]), torch.cat([i_nodes, u_nodes])])
            self.edge_index[behavior] = edge_index

            all_rows.extend(rows)
            all_cols.extend(cols)

        u_nodes = torch.LongTensor(all_rows)
        i_nodes = torch.LongTensor(all_cols) + self.user_count + 1
        self.all_edge_index = torch.stack([torch.cat([u_nodes, i_nodes]), torch.cat([i_nodes, u_nodes])])

    def get_train_loader(self, batch_size):
        dataset = MultiBehaviorTrainDataset(self.user_count, self.item_count, self.train_behavior_dict,
                                            self.behaviors, self.behavior_history)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    def get_val_loader(self, batch_size=None):
        users = [u for u, i in self.validation_interacts.items() if len(i) > 0]
        dataset = TestDataset(users)
        return DataLoader(dataset, batch_size=batch_size or self.args.test_batch_size, shuffle=False)

    def get_test_loader(self, batch_size=None):
        users = [u for u, i in self.test_interacts.items() if len(i) > 0]
        dataset = TestDataset(users)
        return DataLoader(dataset, batch_size=batch_size or self.args.test_batch_size, shuffle=False)