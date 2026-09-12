from fxrl import FXRL
from utils import plot_fx_results, linear_schedule

class TB_plot_study:
    def __init__(self, hyperparameter_name: str, search_space: list, n_iterations: int):
        self.hyperparameter_name = hyperparameter_name
        self.search_space = search_space
        self.n_iterations = n_iterations

        env_kwargs = {
            "order_size": 300,
            "initial_balance": 1000,
            "max_hold_time": 50,  # after what number of timesteps without buy or sell action will agent get penalty
            "penalty_q": 0.0,  # what is starting value of the penalty
            "penalty_k": 0.0,  # how much will the penalty increase every timestep
            "reward_history": 50,  # from how many x number of last timesteps to calculate a reward,
            "reward_scaling": 1e-3
        }
        self.hyperparameters = {
            "learning_rate": linear_schedule(5e-4),
            "n_steps": 1024,
            "batch_size": 1024*8,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "policy_kwargs": None,
            "seed": None,
        }
        self.fxrl = FXRL(env_kwargs=env_kwargs, multiprocess=True, obs_rew_normalization=True,
                         tb_path="./tb")

    def run(self):
        for par_value in self.search_space:
            for n in range(self.n_iterations):
                self.hyperparameters[self.hyperparameter_name] = par_value
                training_name = f"{self.hyperparameter_name}_{par_value}__{n}"
                model = self.fxrl.training(
                    timesteps=2000000,
                    hyper_params=self.hyperparameters,
                    logdir=training_name,
                    use_eval_cb=False,
                    eval_cb_freq=20000
                )
                av, a, _, r = self.fxrl.evaluation(model, IS_data=True)
                print("number of actions: ", a.sum())
                print("cumulative rewards: ", r)
                plot_fx_results(av)

                #av, a, _ = self.fxrl.evaluation(model)
                #print("number of actions: ", a.sum())
                #plot_fx_results(av)


if __name__ == "__main__":
    TB_plot_study(
        hyperparameter_name="clip_range",
        search_space= [0.2,0.2,0.2,0.2],
        n_iterations=1
    ).run()
