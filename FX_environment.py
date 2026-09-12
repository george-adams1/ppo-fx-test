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
    def __init__(self, df, train, order_size, initial_balance, state_features, reward_history, reward_type='percentage',
                 max_hold_time=50, penalty_q=0, penalty_k=0):
        super(ForexTradingEnv, self).__init__()

        self.order_size = order_size
        self.max_hold_time = max_hold_time
        self.initial_balance = initial_balance
        self.state_features = state_features
        self.penalty_q = penalty_q
        self.penalty_k = penalty_k
        self.reward_history = reward_history

        if train == False:
            # don't use penalty in evaluation
            self.penalty_q, self.penalty_k = 0, 0

        self.num_currencies = 3  # Trading 3 currency pairs
        self.reward_type = reward_type

        self.master_data = df.to_numpy().reshape(
            len(df.datetime.unique()),     #number of days
            self.num_currencies,           #number of stocks
            len(df.columns)                #number of columns
        )

        self.name_order_dic = {name: order for order, name in
                               enumerate(df.columns)}  # dict that contains index information about columns in df

        self.sell_indicator = [0 for _ in range(self.num_currencies)]
        self.buy_prices = {i: {} for i in range(self.num_currencies)}

        obs_shape = (1 + self.num_currencies * 3 + self.num_currencies * len(self.state_features))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,))

        self.action_space = spaces.MultiDiscrete([3] * self.num_currencies)  # Discrete actions: hold, buy, sell
        print("obs shape: ", self.observation_space.shape)
        print("action space: ", self.action_space)

        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        self.sell_indicator = [0 for _ in range(self.num_currencies)]

        # Execute actions
        for i in range(self.num_currencies):
            if action[i] == 1:  # Buy
                if self.account_balance >= self.order_size and self.portfolio[i] == 0 and self.candle_data[i][
                    self.name_order_dic["Close"]] != 0:
                    self.account_balance -= self.order_size
                    pcs_bought = self.order_size / self.candle_data[i][self.name_order_dic["Close"]]
                    self.portfolio[i] = pcs_bought
                    self.buy_prices[i][self.current_step] = self.candle_data[i][
                        self.name_order_dic["Close"]]  # Store the buying price
                    self.donothing_time[i] = 0
                else: action[i] = 0
            elif action[i] == 2 :  # Sell
                if self.portfolio[i] > 0 and self.candle_data[i][self.name_order_dic["Close"]] != 0:
                    self.account_balance += self.candle_data[i][self.name_order_dic["Close"]] * self.portfolio[i]
                    self.sell_indicator[i] = 1  # Set sell_indicator[i] to 1 when selling
                    self.portfolio[i] = 0
                    self.donothing_time[i] = 0
                else: action[i] = 0

        self.actions_memory.append(action)

        # new candle ========================================================================================
        self.current_step += 1
        self.candle_data = self.master_data[self.current_step]  # contains data for specific day
        self.total_asset = self.account_balance + sum(
            [self.candle_data[i][self.name_order_dic["Close"]] * owned for i, owned in enumerate(self.portfolio)])
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
        self.rewards_memory.append(self.reward)

        return self._get_observation(), self.calculate_reward(), self.done, {}

    def reset(self):
        self.account_balance = self.initial_balance
        self.portfolio = [0.0 for i in range(self.num_currencies)]
        self.donothing_time = [0.0 for i in range(self.num_currencies)]
        self.current_step = 0
        self.done = False
        self.reward = 0
        self.total_asset = 0

        self.candle_data = self.master_data[0]  # contains data for specific day

        # records
        self.asset_memory = [self.initial_balance]
        self.cash_memory = [self.initial_balance]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.sell_indicator = [0 for _ in range(self.num_currencies)]

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

    def calculate_reward(self, risk_free_rate=0, target_return=0):
        if self.reward_type == 'sortino':
            # As Sortino ratio
            account_value = self.asset_memory[-self.reward_history:]
            returns = np.diff(account_value) / account_value[:-1]
            excess_returns = returns - risk_free_rate
            downside_returns = np.minimum(excess_returns, 0)  # Only consider negative returns
            downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))

            if downside_deviation == 0:
                sortino_ratio = (np.mean(excess_returns) - target_return)
            else:
                sortino_ratio = (np.mean(excess_returns) - target_return) / downside_deviation
            self.reward = sortino_ratio - self.penalty

        elif self.reward_type == 'percentage':

            # Percentage profit calculation

            percentage_profits = []

            for i in range(self.num_currencies):
                if self.sell_indicator[i] and self.buy_prices[i]:
                    transaction_key = list(self.buy_prices[i].keys())[0]  # Get the first transaction key
                    buy_price = self.buy_prices[i][transaction_key]
                    percentage_profit = (self.asset_memory[-1] - buy_price) / buy_price
                    percentage_profits.append(percentage_profit)
                    del self.buy_prices[i][transaction_key]  # Remove the entry from buy_prices
            percentage_profit = sum(percentage_profits) / len(percentage_profits) if percentage_profits else 0
            self.reward = percentage_profit - self.penalty

        return self.reward

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