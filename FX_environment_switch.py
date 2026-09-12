import gym
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from copy import deepcopy
from stable_baselines3.common.env_checker import check_env
from utils import scale_df
import multiprocessing


class ForexTradingEnv(gym.Env):
    def __init__(self, df, train, order_size, initial_balance, state_features, reward_history,
                 max_hold_time=50, penalty_q=0, penalty_k=0, reward_scaling=1):
        super(ForexTradingEnv, self).__init__()

        self.order_size = order_size
        self.max_hold_time = max_hold_time
        self.initial_balance = initial_balance
        self.state_features = state_features
        self.penalty_q = penalty_q
        self.penalty_k = penalty_k
        self.reward_history = reward_history
        self.reward_scaling = reward_scaling
        self.num_currencies = 3  # Trading 3 currency pairs

        if train == False:
            # don't use penalty in evaluation
            self.penalty_q, self.penalty_k = 0, 0

        self.master_data = df.to_numpy().reshape(
            len(df.datetime.unique()),  # number of days
            self.num_currencies,  # number of stocks
            len(df.columns)  # number of columns
        )
        self.name_order_dic = {name: order for order, name in
                               enumerate(df.columns)}  # dict that contains index information about columns in df

        obs_shape = (1 + self.num_currencies * 3 + self.num_currencies * len(self.state_features))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,))
        self.action_space = spaces.MultiDiscrete([2] * self.num_currencies)  # Discrete actions: hold, buy, sell
        print("obs shape: ", self.observation_space.shape)
        print("action space: ", self.action_space)

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        # Execute actions
        for i in range(self.num_currencies):
            if action[i] == 1: #makes a switch
                price = self.candle_data[i][self.name_order_dic["Close"]]
                if price != 0:

                    if self.portfolio[i] <= 0:
                        if self.portfolio[i] < 0: #the first timestep agent has nothing borrowed
                            # end shorting
                            self.account_balance -= -(self.portfolio[i]) * price
                            self.portfolio[i] = 0

                            self.value_borrowed[i] = 0

                        # buying
                        self.account_balance -= self.order_size
                        self.portfolio[i] = self.order_size / price

                        self.donothing_time[i] = 0
                        self.trade_done[i] = True

                    elif self.portfolio[i] > 0:
                        # selling
                        self.account_balance += self.portfolio[i] * price
                        self.portfolio[i] = 0
                        # start shorting
                        self.account_balance += self.order_size
                        self.portfolio[i] = -(self.order_size / price)

                        self.value_borrowed[i] = self.order_size
                        self.donothing_time[i] = 0
                        self.trade_done[i] = True

                else:
                    action[i] = 0

        self.actions_memory.append(action)

        # new candle ========================================================================================
        self.current_step += 1
        self.candle_data = self.master_data[self.current_step]  # contains data for specific day
        self.total_asset = self.account_balance - sum(self.value_borrowed) + sum([self.candle_data[i][self.name_order_dic["Close"]]*owned for i, owned in enumerate(self.portfolio) if owned > 0])

        # Check if the episode is done
        if self.current_step == self.master_data.shape[0] - 1 or self.total_asset < self.order_size:
            self.done = True

        # Update hold time
        self.donothing_time = [i + 1 for i in self.donothing_time]

        # Calculate penalties
        self.penalty = 0
        for i in range(self.num_currencies):
            if self.donothing_time[i] > self.max_hold_time:
                self.penalty += self.donothing_time[i] * self.penalty_k + self.penalty_q

        self.asset_memory.append(self.total_asset)
        self.cash_memory.append(self.account_balance)
        self.date_memory.append(self._get_date())
        reward = self.calculate_reward()
        self.total_reward += reward

        return self._get_observation(), reward, self.done, {}

    def reset(self):
        self.account_balance = self.initial_balance
        self.portfolio = [0.0 for i in range(self.num_currencies)]
        self.donothing_time = [0.0 for i in range(self.num_currencies)]
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_asset = 0

        self.candle_data = self.master_data[0]  # contains data for specific day

        self.previous_price = [0 for _ in range(self.num_currencies)] #price at which forex/stocks was bought/shorted
        self.trade_done = [False for _ in range(self.num_currencies)] #True if order was executed at certain timestep
        self.value_borrowed = [0 for _ in range(self.num_currencies)] #value of forex/stocks borrowed for shorting

        # records
        self.asset_memory = [self.initial_balance]
        self.cash_memory = [self.initial_balance]
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.total_reward = 0

        return self._get_observation()

    def _get_observation(self):
        obs = np.array(
            [self.account_balance] +
            [subarray[self.name_order_dic["Close"]] for subarray in self.candle_data] +
            self.portfolio +
            self.donothing_time +
            sum([subarray[[self.name_order_dic[key] for key in self.state_features]].tolist() for subarray in
                 self.candle_data], [])
        )
        return obs

    def calculate_reward(self):

        def calculate_change(old_value, new_value, shorting):
            if old_value == 0:
                change = 0
            else:
                change = ((new_value - old_value) / old_value)
            if shorting:
                change = change * -1
            return change

        reward = 0
        for i in range(self.num_currencies):
            if self.trade_done[i]:
                current_price = self.master_data[self.current_step-1][i][self.name_order_dic["Close"]]
                shorting = True if self.portfolio[i] > 0 else False
                reward += calculate_change(self.previous_price[i], current_price, shorting)
                self.trade_done[i] = False
                self.previous_price[i] = current_price
        return reward * self.reward_scaling

    def _get_date(self):
        date = self.candle_data[0][self.name_order_dic["datetime"]]
        return date

    def get_records(self):
        date_list = self.date_memory

        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )

        cash_list = self.cash_memory
        df_cash = pd.DataFrame(
            {"date": date_list, "money in cash": cash_list}
        )

        if self.num_currencies > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = [self.candle_data[i][self.name_order_dic["pair"]] for i in
                                  range(len(self.candle_data))]
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})

        return df_account_value, df_actions, df_cash

    def get_sb_env(self):
        e = DummyVecEnv([lambda: Monitor(deepcopy(self))])
        obs = e.reset()
        return e, obs

    def get_multiproc_env(self, n=multiprocessing.cpu_count()):
        def get_self():
            return Monitor(deepcopy(self))

        e = SubprocVecEnv([get_self for _ in range(n)], start_method="spawn")
        obs = e.reset()
        return e, obs


if __name__ == '__main__':
    # check environment with SB3's checker
    df = pd.read_csv("fx_data/full_df.csv", index_col=[0])
    df = scale_df(df)
    state_features = [c for c in df.columns.tolist() if c not in ["Close", "datetime", "pair"]]
    kwargs = {
        "order_size": 300,
        "initial_balance": 1000,
        "reward_history": 100,  # from how many x number of last timesteps to calculate a reward
        "state_features": state_features
    }
    env = ForexTradingEnv(df=df, train=False, **kwargs)
    check_env(env)