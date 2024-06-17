import torch.nn as nn


class config_thirteen:
    def __init__(self, use_baseline, ppo, seed):
        self.env_name = "CardGame-v0"
        self.record = False
        baseline_str = ("baseline" if use_baseline else "no_baseline") if not ppo else "ppo"
        seed_str = "seed=" + str(seed)

        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, baseline_str, seed_str
        )
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # model and training config1
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 10000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 1e-3
        self.gamma = 1.0  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 64

        # hyperparameters for PPO
        self.eps_clip = 0.2
        self.update_freq = 20

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


def get_config(env_name, baseline, ppo, seed=15):
    if env_name == "CardGame-v0":
        return config_thirteen(baseline, ppo, seed)

