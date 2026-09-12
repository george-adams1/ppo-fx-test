import time
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from FX_environment_switch import ForexTradingEnv
from utils import scale_df, create_env, linear_schedule, plot_fx_results
import warnings
import os
from stable_baselines3.common.vec_env import VecNormalize
warnings.simplefilter(action='ignore', category=FutureWarning)


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = 10000
        self.path = "./models"

    def _init_callback(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        self.model.save(os.path.join(self.path, "model_step_0"))

    def _on_step(self) -> bool:
        rewards = self.training_env.get_attr("total_reward")
        step = self.training_env.get_attr("current_step")
        self.logger.record("custom/rewards", np.mean(rewards))
        self.logger.record("custom/steps", np.mean(step))

        #every x number of timesteps model will be saved
        if self.n_calls % self.save_freq == 0:
            self.model.save(os.path.join(self.path, f"model_step_{self.num_timesteps}"))
        return True


class FXRL:
    def __init__(self, env_kwargs, multiprocess, obs_rew_normalization, tb_path):
        self._data_prep()
        self.tb_path = tb_path

        # create envs
        print(len(self.data_IS.datetime.unique()))
        self.eval_env = ForexTradingEnv(df=self.data_OOS, train=False, state_features=self.state_features, **env_kwargs)
        self.train_env = ForexTradingEnv(df=self.data_IS, train=True, state_features=self.state_features, **env_kwargs)
        self.train_env_sb = create_env(self.train_env, multiprocess=multiprocess, obs_rew_normalization=obs_rew_normalization)
        self.train_env_sb.save('vec_normalize.zip')

        self.IS_env_test, _ = self.train_env.get_multiproc_env()
        self.IS_env_test = VecNormalize(self.IS_env_test, training=False, norm_obs=True, norm_reward=True)


    def _data_prep(self, date_split="2023-03-01"):
        df = pd.read_csv("fx_data/full_df.csv", index_col=[0])
        df = scale_df(df, feature_to_skip=["pair", "datetime", "Close"])

        self.state_features = [c for c in df.columns.tolist() if c not in ["Close", "datetime", "pair"]]
        self.data_IS = df[df.datetime < date_split]
        self.data_OOS = df[df.datetime > date_split]

    def _add_callbacks(self, use_eval_cb, eval_cb_freq):
        callbacks = []
        if use_eval_cb:
            eval_callback = EvalCallback(self.IS_env_test, #should be eval_env!!!, for now temporary train env
                                         best_model_save_path='./best_model',
                                         log_path='./tb',
                                         eval_freq=eval_cb_freq,  # Evaluate the model every X timesteps
                                         deterministic=True,
                                         render=False)
            callbacks.append(eval_callback)
        callbacks.append(CustomCallback())
        return callbacks

    def training(self, timesteps, hyper_params, logdir, use_eval_cb, eval_cb_freq):
        callbacks = self._add_callbacks(use_eval_cb, eval_cb_freq)

        start_time = time.time()

        model = PPO("MlpPolicy", self.train_env_sb, verbose=1, tensorboard_log="./tb", device='cuda', **hyper_params)

        end_time = time.time()

        trained_model = model.learn(total_timesteps=timesteps, tb_log_name=logdir, callback=callbacks)

        elapsed_time = end_time - start_time
        print("Training took {:.2f} seconds".format(elapsed_time))

        return trained_model

    def evaluation(self, model, deterministic=True, IS_data=False):
        if IS_data:
            env = VecNormalize(self.train_env, training=False)
        else:
            env = VecNormalize(self.eval_env, training=False)

        env = VecNormalize.load('vec_normalize.zip', env)

        total_rewards = 0
        test_obs = env.reset()
        for i in range(len(env.master_data)):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, done, info = env.step(action)
            total_rewards += rewards
            if i == (len(env.master_data) - 3):
                df_account_value, df_actions, df_cash = env.get_records()
                break
            if done:
                df_account_value, df_actions, df_cash = env.get_records()
                print("ups, early crash!")
                break
        return df_account_value, df_actions, df_cash, total_rewards



if __name__ == "__main__":
    env_kwargs = {
        "order_size": 300,
        "initial_balance": 1000,
        "max_hold_time": 50,  # after what number of timesteps without buy or sell action will agent get penalty
        "penalty_q": 0.0,  # what is starting value of the penalty
        "penalty_k": 0.0,  # how much will the penalty increase every timestep
        "reward_history": 50,  # from how many x number of last timesteps to calculate a reward,
        "reward_scaling": 1e-3
    }

    hyperparameters = {
        "learning_rate": linear_schedule(5e-3),
        "n_steps": 1024,
        "batch_size": 1024,
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

    fxrl = FXRL(env_kwargs=env_kwargs, multiprocess=True, obs_rew_normalization=True, tb_path="./tb")

    model = fxrl.training(
        timesteps=50000000,
        hyper_params=hyperparameters,
        logdir="SavingModel",
        use_eval_cb=True,
        eval_cb_freq=10000
    )

    # # eval on OS data
    # av, a, _, r = fxrl.evaluation(model)
    # print("number of actions: ", a.sum())
    # plot_fx_results(av)

    # eval on IS data
    # av, a, _, r = fxrl.evaluation(model, IS_data=True)
    # print("number of actions: ", a.sum())
    # plot_fx_results(av)