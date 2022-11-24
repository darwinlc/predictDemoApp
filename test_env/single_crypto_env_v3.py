import numpy as np
import os
import gym
from numpy import random as rd
import math
from numpy import fft

# Tools
def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))


def cos_sum(a, b):
    """To work with tabulate."""
    return math.cos(a + b)


class GAF:
    def __init__(self):
        pass

    def __call__(self, serie):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2 * serie - max_ - min_) / (max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1.0, 1.0, scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1.0, -1.0, scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)
        # Note! The computation of r is not necessary
        r = np.linspace(0, 1, len(scaled_serie))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        return (gaf, phi, r, scaled_serie)


def fourierExtrapolation(x, n_predict, n_harm=5):
    n = x.size
    # n_harm = 5                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    x_notrend = x - p[0] * t  # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[: 1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


class SingleStockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        gamma=0.99,
        min_stock_rate=0.0001,
        buy_min_value=10.0,
        sell_min_value=10.0,
        cash_min_value=1.0,
        stock_min_value=1.0,
        initial_capital=1e6,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        reward_scaling=2 ** -11,
        initial_stocks=None,
        max_step=30,
    ):
        # price_ary: open, close, low, high
        price_ary = config["price_array"]
        train_idx_list = config["train_start_idx"]
        if_test = config["if_test"]

        # reward based on value or coin count
        if_value = config["if_value"]
        # lookback px history; default 30; to predict using FFT
        self.lookback_n = config.get("lookback_n", 30)
        # GAF graph (v1, v2): v1: history, v2: fft prediction
        self.gaf_dim = config.get("gaf_dim", (15, 5))

        # time duration
        n = price_ary.shape[0]
        self.price_ary = price_ary.astype(np.float32)

        # single crypto
        stock_dim = 1
        self.gamma = gamma
        self.buy_min_value = buy_min_value
        self.sell_min_value = sell_min_value
        self.cash_min_value = cash_min_value
        self.stock_min_value = stock_min_value
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.run_index = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "SingleTradeEnv"

        # cash + cash/(coin_value + cash) + GAF dimension
        self.gaf = GAF()
        self.gaf_dim_sum = self.gaf_dim[0] + self.gaf_dim[1]
        # self.state_dim = 2 * self.gaf_dim_sum + self.gaf_dim_sum * self.gaf_dim_sum
        self.state_dim = self.gaf_dim_sum + self.gaf_dim_sum * self.gaf_dim_sum

        print("State Dim: ", self.state_dim)

        self.action_dim = stock_dim

        # max game duration
        self.max_step = max_step
        self.max_datalength = n - 1
        self.if_test = if_test
        self.if_value = if_value
        self.if_discrete = False

        # flag to check whether any trade is done
        self.flag_trade = 0.0
        self.train_idx_list = train_idx_list

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-0.9, high=0.9, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        if not self.if_test:
            # initiate portfolio for training
            self.stocks = self.initial_stocks.copy()
            # add noise
            self.amount = self.initial_capital * (rd.rand() * 0.1 + 0.8)
            # random start idx
            self.day = np.random.choice(self.train_idx_list, 1)[0]
            # print(f'Random at {self.day}')
            self.run_index = 0
            price = self.price_ary[self.day]
        else:
            # for testing
            assert len(self.train_idx_list) == 1
            self.stocks = self.initial_stocks.astype(np.float32)
            self.amount = self.initial_capital
            self.day = self.train_idx_list[0]
            self.run_index = 0
            price = self.price_ary[self.day]

        self.total_asset = self.amount + (self.stocks[0] * price[1])
        self.initial_total_asset = self.total_asset

        self.hist_state = self.stocks * np.mean(self.price_ary[:, 2])
        self.episode_return = 0.0
        self.gamma_reward = 0.0
        # count number of trades
        self.flag_trade = 0.0

        init_state = self.get_state(price)
        return init_state

    def step(self, actions):
        def tradable_size(x):
            return (x / self.min_stock_rate).astype(int) * self.min_stock_rate

        # actions -> percentage of stock or cash
        actions_v = actions[0]
        # prx_v = actions[1]

        if actions_v == np.nan:
            actions_v = 0.0

        # version 0: order price -> last day (open + close)/2; order can be filled quickly
        order_px = (
            self.price_ary[self.day + self.run_index, 0]
            + self.price_ary[self.day + self.run_index, 1]
        ) / 2.0

        # version 1: last closing price * action dimesion 2
        # order_px = self.price_ary[self.day + self.run_index, 1] * (1.0 + prx_v)

        self.run_index += 1
        price = self.price_ary[self.day + self.run_index]

        # within day low-high
        if actions_v > 0:
            if self.amount * actions_v > self.buy_min_value:
                buy_num_shares = tradable_size(
                    (self.amount * actions_v / order_px) / (1 + self.buy_cost_pct)
                )

                flag_order_place = False
                if self.if_test and buy_num_shares != 0.0:
                    flag_order_place = True
                    print(f"[Day {self.day + self.run_index}] BUY: {buy_num_shares}")

                if order_px > price[2]:
                    if self.if_test and flag_order_place:
                        print("Order filled")
                    actual_order_px = min(order_px, price[3])
                    self.stocks[0] += buy_num_shares
                    amt_delta = (
                        actual_order_px * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.amount -= amt_delta
                    self.hist_state += amt_delta
                    self.flag_trade += 1
                else:
                    if self.if_test and flag_order_place:
                        print("Order expired")

        if actions_v < 0:
            sell_num_shares = tradable_size(self.stocks[0] * (-1.0) * actions_v)
            # no short
            sell_num_shares = min(
                sell_num_shares, self.stocks[0] - self.stock_min_value
            )

            if (order_px * sell_num_shares) > self.sell_min_value:

                flag_order_place = False
                if self.if_test and sell_num_shares != 0.0:
                    flag_order_place = True
                    print(f"[Day {self.day + self.run_index}] SELL: {sell_num_shares}")

                if order_px < price[3]:
                    if self.if_test and flag_order_place:
                        print("Order filled")
                    actual_order_px = max(order_px, price[2])
                    self.stocks[0] = self.stocks[0] - sell_num_shares
                    amt_delta = (
                        actual_order_px * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.amount += amt_delta
                    self.hist_state -= amt_delta
                    self.flag_trade += 1
                else:
                    if self.if_test and flag_order_place:
                        print("Order expired")
        # State
        state = self.get_state(price)
        # print (state)
        total_asset = self.amount + (self.stocks[0] * price[1])
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset
        self.gamma_reward = self.gamma_reward * self.gamma + reward

        done = (self.run_index == self.max_step) or (
            (self.run_index + self.day) >= (self.price_ary.shape[0] - 1)
        )
        if self.flag_trade > 0:
            # if model sold out all stocks
            done = done or (self.stocks[0] == self.stock_min_value)

        if done:
            reward = self.gamma_reward
            self.episode_return = self.total_asset / self.initial_total_asset

            if self.if_test:
                print(
                    "Episode Return: \t",
                    self.episode_return,
                    "\t Number of Trade: \t",
                    self.flag_trade,
                    "\t Duration of Trade: \t",
                    self.run_index,
                )
            # self.reset()

        return state, reward, done, dict()

    def get_state(self, price):
        # current cash rati
        cash_ratio = np.array(
            self.amount / (self.amount + (self.stocks * price[1]).sum() + 1e-10),
            dtype=np.float32,
        )
        # normalized cash by comparing to initial capital (default weight at 2.0)
        norm_cash = np.array(
            self.amount / (2.0 * self.initial_capital), dtype=np.float32
        )

        # predict prx
        px_index_st = max(0, self.day + self.run_index - self.lookback_n + 1)
        px_index_ed = self.day + self.run_index + 1
        fft_input_price = self.price_ary[px_index_st:px_index_ed, 1]
        fft_extra = fourierExtrapolation(fft_input_price, self.gaf_dim[1])

        # encoding to GAF
        new_price = np.hstack(
            (
                fft_input_price[(-1 * self.gaf_dim[0]) :],
                fft_extra[(-1 * self.gaf_dim[1]) :],
            )
        )
        g, _, _, _ = self.gaf(new_price)

        return np.hstack(
            (
                np.tile(norm_cash, self.gaf_dim_sum),
                np.tile(cash_ratio, self.gaf_dim_sum),
                g.flatten(),
            )
        )


    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
