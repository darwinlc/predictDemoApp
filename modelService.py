from shutil import register_unpack_format
import numpy as np
from datetime import datetime, timedelta
from neo_finrl.data_processors.processor_yahoofinance import YahooFinanceProcessor

from test_env.single_crypto_env_v3 import SingleStockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

# RL model parameters
reward_on_value = True
lookback_n = 30

config_max_step = 120

if reward_on_value:
    reward_scaling = 2 ** -3
else:
    reward_scaling = 2 ** -3


def ETL(stock_name):
    stock_tic_mapper = {"BTC": "BTC-USD", "CMRE": "CMRE"}
    tic_list = [stock_tic_mapper.get(stock_name)]
    today_d = datetime.today()
    start_d = today_d - timedelta(days=40)

    query_start = start_d.strftime("%Y-%m-%d")
    query_end = today_d.strftime("%Y-%m-%d")
    print(query_start, query_end)

    tech_indicators = ["rsi_2"]
    data_downloader = YahooFinanceProcessor()
    stock_history_df = data_downloader.download_data(
        query_start, query_end, tic_list, "1D"
    )

    flag_extract_fail = stock_history_df.shape[0] == 0

    if not flag_extract_fail:
        data_downloader.time_interval = "1D"
        stock_history_df = data_downloader.clean_data(stock_history_df)
        stock_history_df = data_downloader.add_technical_indicator(
            stock_history_df, tech_indicators
        )
        stock_history_df["rsi_2-3"] = stock_history_df["rsi_2"].shift(
            3, fill_value=50.0
        )

        # test game start signal
        rsi_2_v = stock_history_df.iloc[-3]["rsi_2"]
        rsi_2_v2 = stock_history_df.iloc[-3]["rsi_2-3"]
        print(f"RSI_2[d-2]: {rsi_2_v}, {rsi_2_v2}")

        flag_gamestart = False
        if (rsi_2_v > 70) & (rsi_2_v2 < 30):
            flag_gamestart = True
        return stock_history_df[["open", "adjcp", "low", "high"]], flag_gamestart
    else:
        return None, None


def getStockConfig(stock_name):
    # security parameters
    # minimal stock unit, minimal amount of selling value, minimal amount of buying value
    # minimal holding cash value, minimal stock quantity
    security_config = {
        "BTC": [0.001, 0.0, 0.0, 10.0, 0.002],
        "CMRE": [1.0, 0.0, 0.0, 10.0, 1.0],
    }
    return security_config.get(stock_name, None)


def modelRun(start_idx, px_df, input_amount, input_stocks, last_model, stock_name):
    def tradable_size(env, x):
        return (x / env.min_stock_rate).astype(int) * env.min_stock_rate

    max_step = min(config_max_step, px_df.shape[0] - start_idx[0]) - 1

    test_config = dict()

    test_config["price_array"] = px_df[["open", "adjcp", "low", "high"]].values
    # randomly start day index for back testing
    test_config["if_test"] = True
    test_config["train_start_idx"] = start_idx

    test_config["if_value"] = reward_on_value
    test_config["lookback_n"] = lookback_n

    print("Run model from ", start_idx[0], " to ", start_idx[0] + max_step)

    sec_config = getStockConfig(stock_name)
    if sec_config != None:
        min_stock_rate = sec_config[0]
        sell_min_value = sec_config[1]
        buy_min_value = sec_config[2]
        cash_min_value = sec_config[3]
        stock_min_value = sec_config[4]
    else:
        return "Missing security configuration Info. Aborted"

    test_env = SingleStockTradingEnv(
        test_config,
        initial_capital=input_amount,
        initial_stocks=input_stocks,
        max_step=max_step,
        reward_scaling=reward_scaling,
        min_stock_rate=min_stock_rate,
        sell_min_value=sell_min_value,
        buy_min_value=buy_min_value,
        cash_min_value=cash_min_value,
        stock_min_value=stock_min_value,
        gamma=0.8,
    )
    state = test_env.reset()

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    test_model = PPO.load(last_model, custom_objects=custom_objects)
    test_model = test_model.policy.eval()

    action = test_model.predict(state)[0]
    # actions -> percentage of stock or cash
    # add clip at 0.9
    actions_v = action[0]

    if actions_v == np.nan:
        actions_v = 0.0

    order_px = (
        test_env.price_ary[test_env.day + test_env.run_index, 0]
        + test_env.price_ary[test_env.day + test_env.run_index, 1]
    ) / 2.0

    if actions_v > 0.1:
        action_msg = "[BUY]"
    elif actions_v < -0.1:
        action_msg = "[SELL]"
    else:
        action_msg = ""

    # 3decimal
    action_v_str = int(actions_v * 1000) / 1000.0
    order_px_str = int(order_px * 1000) / 1000.0
    price_msg = f"Suggested Price: {order_px_str}"

    return f"Action value: {action_v_str} {action_msg}\n{price_msg}"

    # if actions_v > 0:
    #     buy_num_shares = tradable_size(
    #         test_env,
    #         (test_env.amount * actions_v / order_px) / (1 + test_env.buy_cost_pct),
    #     )
    #     if buy_num_shares > 0:
    #         print(f"Buy {buy_num_shares} at price {order_px}")
    #     else:
    #         print("Suggest to buy, but balance not available to trade today")

    # if actions_v < 0:
    #     sell_num_shares = tradable_size(
    #         test_env, test_env.stocks[0] * (-1.0) * actions_v
    #     )
    #     # no short
    #     sell_num_shares = min(sell_num_shares, test_env.stocks[0])
    #     if sell_num_shares > 0:
    #         print(f"Sell {sell_num_shares} at price {order_px}")
    #     else:
    #         print("Suggest to sell, but balance not available to trade today")

    # print("\n")
    # print(
    #     "[!!Warning!!] Order may not be able to placed if it is lower the mininal trade amount!!"
    # )
    # print("[!!Warning!!] check current MKT price for better deal!!")


def getModelFile(stock_name):
    model_mapping = {"BTC": "./model/BTC_model.zip", "CMRE": "./model/CMRE_model.zip"}
    return model_mapping.get(stock_name, None)


def run(stock_name, current_q, current_c):
    # check model file
    model_file = getModelFile(stock_name)
    # etl
    px_df, flag_start = ETL(stock_name)
    if (flag_start != None) and (model_file != None):
        if flag_start:
            msg_trade_start = "New Entry Signal: On"
        else:
            msg_trade_start = "New Entry Signal: Off"

        model_result = modelRun(
            [px_df.shape[0] - 1], px_df, current_c, current_q, model_file, stock_name
        )
        return "\n".join([model_result, msg_trade_start])
    else:
        return "Error when loading data/model"


def webcall(stock_name, current_q, current_c):
    if type(current_q) != float:
        current_q = float(current_q)
    if type(current_c) != float:
        current_c = float(current_c)

    ret_v = run(stock_name, np.array([current_q]), current_c)
    return ret_v


def test_func():
    v = run("BTC", np.array([0.1]), 1000.0)
    print(v)


if __name__ == "__main__":
    test_func()
