import logging

import numpy as np

from .client import HFLClient
from .group import Group
from ..fedavg.fedavg_api import FedAvgAPI
import ast


class HierarchicalTrainer(FedAvgAPI):
    def _setup_clients(
            self,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        if self.args.group_method == "random":
            self.group_indexes = np.random.randint(
                0, self.args.group_num, self.args.client_num_in_total
            )
            group_to_client_indexes = {}
            for client_idx, group_idx in enumerate(self.group_indexes):
                if not group_idx in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
        elif self.args.group_method == "custom":
            # 解析用户传入的字符串格式的分组定义
            try:
                # 将字符串解析为字典
                group_to_client_indexes = ast.literal_eval(self.args.custom_group_str)
                # 验证字典的结构是否正确
                if not isinstance(group_to_client_indexes, dict):
                    raise ValueError("custom_group_str must be a dictionary")
                for group_idx, client_list in group_to_client_indexes.items():
                    if not isinstance(group_idx, int):
                        raise ValueError("Group index must be an integer")
                    if not isinstance(client_list, list):
                        raise ValueError("Client indexes must be a list")
                    for client_idx in client_list:
                        if not isinstance(client_idx, int):
                            raise ValueError("Client index must be an integer")
            except Exception as e:
                raise ValueError(f"Invalid custom_group_str format: {e}")
        else:
            raise Exception(self.args.group_method)

        self.group_dict = {}
        for group_idx, client_indexes in group_to_client_indexes.items():
            self.group_dict[group_idx] = Group(
                group_idx,
                client_indexes,
                train_data_local_dict,
                test_data_local_dict,
                train_data_local_num_dict,
                self.args,
                self.device,
                self.model,
                self.model_trainer
            )

        # maintain a dummy client to be used in FedAvgTrainer::local_test_on_all_clients()
        client_idx = -1
        self.client_list = [
            HFLClient(
                client_idx,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                self.model,
                self.model_trainer
            )
        ]
        logging.info("############setup_clients (END)#############")

    def _client_sampling(self, global_round_idx, client_num_in_total, client_num_per_round):
        if self.args.group_method == "custom":
            # 如果是自定义分组，直接使用自定义的分组
            group_to_client_indexes = ast.literal_eval(self.args.custom_group_str)
            # 验证字典的结构是否正确
            if not isinstance(group_to_client_indexes, dict):
                raise ValueError("custom_group_str must be a dictionary")
            for group_idx, client_list in group_to_client_indexes.items():
                if not isinstance(group_idx, int):
                    raise ValueError("Group index must be an integer")
                if not isinstance(client_list, list):
                    raise ValueError("Client indexes must be a list")
                for client_idx in client_list:
                    if not isinstance(client_idx, int):
                        raise ValueError("Client index must be an integer")
            logging.info(
                "client_indexes of each group = {}".format(group_to_client_indexes)
            )
            return group_to_client_indexes
        else:
            # 随机采样逻辑保持不变
            sampled_client_indexes = super()._client_sampling(
                global_round_idx, client_num_in_total, client_num_per_round
            )
            group_to_client_indexes = {}
            for client_idx in sampled_client_indexes:
                group_idx = self.group_indexes[client_idx]
                if group_idx not in group_to_client_indexes:
                    group_to_client_indexes[group_idx] = []
                group_to_client_indexes[group_idx].append(client_idx)
            logging.info(
                "client_indexes of each group = {}".format(group_to_client_indexes)
            )
            return group_to_client_indexes

    def train(self):
        acc_list = []
        loss_list = []
        w_global = self.model.state_dict()
        for global_round_idx in range(self.args.comm_round):
            test_acc = 0
            test_loss = 0
            logging.info(
                "################Global Communication Round : {}".format(
                    global_round_idx
                )
            )
            group_to_client_indexes = self._client_sampling(
                global_round_idx,
                self.args.client_num_in_total,
                self.args.client_num_per_round,
            )

            # train each group
            w_groups_dict = {}
            for group_idx in sorted(group_to_client_indexes.keys()):
                sampled_client_indexes = group_to_client_indexes[group_idx]
                group = self.group_dict[group_idx]
                w_group_list = group.train(
                    global_round_idx, w_global, sampled_client_indexes
                )
                for global_epoch, w in w_group_list:
                    if not global_epoch in w_groups_dict:
                        w_groups_dict[global_epoch] = []
                    w_groups_dict[global_epoch].append(
                        (group.get_sample_number(sampled_client_indexes), w)
                    )

            # aggregate group weights into the global weight
            for global_epoch in sorted(w_groups_dict.keys()):
                w_groups = w_groups_dict[global_epoch]
                w_global = self._aggregate(w_groups)

                # evaluate performance
                if (
                        global_epoch % self.args.frequency_of_the_test == 0
                        or global_epoch
                        == self.args.comm_round
                        * self.args.group_comm_round
                        * self.args.epochs
                        - 1
                ):
                    self.model.load_state_dict(w_global)
                    test_acc, test_loss = self._local_test_on_all_clients(global_epoch)

            if test_acc != 0:
                acc_list.append(test_acc)
                loss_list.append(test_loss)

        return acc_list, loss_list

