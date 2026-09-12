from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.vec_env import VecNormalize
import pandas as pd
import matplotlib.pyplot as plt


def scale_df(df, feature_to_skip: list = ["pair", "datetime", "Close"]):
    sf_list = [c for c in df.columns if c not in feature_to_skip]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[sf_list])
    scaled_df = pd.DataFrame(scaled_features, columns=sf_list)
    df_scaled = pd.concat([df.drop(sf_list, axis=1), scaled_df], axis=1)
    return df_scaled


def create_env(env, multiprocess=False, obs_rew_normalization=False):

    if multiprocess:
        # multi-processing
        e, _ = env.get_multiproc_env()
    else:
        # single-processing
        e, _ = env.get_sb_env()

    if obs_rew_normalization:
        e = VecNormalize(e, training=True, norm_obs=True, norm_reward=True)
    return e

def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def plot_fx_results(df_account_value):
    df_forex_pairs = pd.read_csv("fx_data/full_df.csv", index_col=[0])
    df_forex_pairs['datetime'] = pd.to_datetime(df_forex_pairs['datetime'])
    df_forex_pairs.set_index("datetime", inplace=True)
    df_account_value['date'] = pd.to_datetime(df_account_value['date'])
    df_account_value.set_index("date", inplace=True)

    account_value_series = df_account_value['account_value']
    start_date = df_account_value.index[0]
    end_date = df_account_value.index[-1]
    df_forex_pairs = df_forex_pairs.loc[start_date:end_date]

    forex_pairs = ["eurusd", "gbpusd", "usdjpy"]
    df_forex_pairs = df_forex_pairs[df_forex_pairs['pair'].isin(forex_pairs)]
    df_forex_pairs['Close'] /= df_forex_pairs.groupby('pair')['Close'].transform(lambda x: x.iloc[0])

    benchmark_series = df_forex_pairs.groupby(df_forex_pairs.index)['Close'].mean()
    benchmark_series *= (account_value_series.iloc[0] / benchmark_series.iloc[0])

    plt.figure(figsize=(12, 6))
    plt.plot(account_value_series.index, account_value_series.values, label='Account Value')
    plt.plot(benchmark_series.index, benchmark_series.values, color='grey', label='Benchmark Index')
    plt.xlabel('Date')
    plt.ylabel('Account Value')
    plt.title('Account Value over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

