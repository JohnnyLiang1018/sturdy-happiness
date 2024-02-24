import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import matplotlib.pyplot as plt

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            replay_buffer_real: ReplayBuffer, ##
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop_sim,
            num_expl_steps_per_train_loop_real,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_frequency=0,

    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop_sim = num_expl_steps_per_train_loop_sim ##
        self.num_expl_steps_per_train_loop_real = num_expl_steps_per_train_loop_real ##
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_frequency = save_frequency
        self.replay_buffer_real = replay_buffer_real ##
        
    def _train(self):
        print("beginning of training")
        initial_collection_request = True
        if self.min_num_steps_before_training > 0:
            print("collect initial path")
            init_expl_paths_sim, init_expl_paths_real = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                self.num_expl_steps_per_train_loop_real,
                discard_incomplete_paths=False,
                collect_real_paths=False,
                initial_collection_request=initial_collection_request,
            )
            initial_collection_request = False
            # init_expl_paths = self.expl_data_collector.collect_new_paths(
            #     self.max_path_length,
            #     self.min_num_steps_before_training,
            #     discard_incomplete_paths=False,
            # )
            self.replay_buffer.add_paths(init_expl_paths_sim)
            self.replay_buffer_real.add_paths(init_expl_paths_real)
            # self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        count = 0 ##
        epoch_num = 0
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            print("Size of real buffer", self.replay_buffer_real.num_steps_can_sample())
            print("Size of sim buffer", self.replay_buffer.num_steps_can_sample())
            print("start eval collector")
            # self.eval_data_collector.collect_new_paths(
            #     self.max_path_length,
            #     self.num_eval_steps_per_epoch,
            #     discard_incomplete_paths=True,
            # )
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                num_steps_real=0,
                discard_incomplete_paths=True,
                collect_real_paths=False,
            )
            # print("end eval collector")
            # gt.stamp('evaluation sampling')
            

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths_sim, new_expl_paths_real = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop_sim,
                    self.num_expl_steps_per_train_loop_real,
                    discard_incomplete_paths=False,
                    collect_real_paths=False,
                    initial_collection_request = initial_collection_request,
                )
                initial_collection_request = False
                # new_expl_paths = self.expl_data_collector.collect_new_paths(
                #     self.max_path_length,
                #     self.num_expl_steps_per_train_loop,
                #     discard_incomplete_paths=False,
                # )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths_sim)
                self.replay_buffer_real.add_paths(new_expl_paths_real)
                # self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                print("start training")

                for _ in range(self.num_trains_per_train_loop):
                    if count > self.num_epochs / 2:
                        train_data_sim = self.replay_buffer.random_batch(self.batch_size)
                        train_data_real = self.replay_buffer_real.random_batch(self.batch_size)
                        # train_data_real = self.replay_buffer_real.random_batch(round(self.batch_size - self.batch_size/3))
                        train_data_sim_ = self.replay_buffer.random_batch(round(self.batch_size/3))
                        tuning = True
                    else:
                        train_data_sim = self.replay_buffer.random_batch(round(self.batch_size - self.batch_size*0.5))
                        # train_data_real_ = self.replay_buffer_real.random_sub_batch(epoch_num*100 + 6000, round(self.batch_size*0.3))
                        train_data_sim_ = self.replay_buffer.random_batch(round(self.batch_size*0.5))
                        train_data_real = self.replay_buffer_real.random_sub_batch(epoch_num*100 + 1000, round(self.batch_size*0.5))
                        # train_data_sim = self.replay_buffer.random_batch(round(self.batch_size - self.batch_size/3))
                        # train_data_real = self.replay_buffer_real.random_batch(round(self.batch_size/3))
                        tuning = False
                    # train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train_exp(train_data_sim, train_data_sim_, train_data_real, tuning, epoch_num, old_appr=False)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            # count += 1

            epoch_num += 1
            self._end_epoch(epoch)
            if self.save_frequency > 0:
                if epoch % self.save_frequency == 0:
                    self.trainer.save_models(epoch)
                    self.replay_buffer.save_buffer(str(epoch)+'_sim')
                    self.replay_buffer_real.save_buffer(str(epoch)+'_real')
