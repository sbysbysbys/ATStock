import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.dates as mdates
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from datasets.akshareutils import get_stock_name

config_path = ".//visulization//vis.yaml"

def vis_daily(symbol=0, graph_start_date=0, graph_type=0):
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    daily_dir = config["daily"]["dir"]
    if graph_start_date == 0:
        graph_start_date = config["daily"]["graph_start_date"]
    if graph_type == 0:
        graph_type = config["daily"]["graph_type"]
    if symbol == 0:
        s_symbol = config["daily"]["symbol"]
    else:
        s_symbol = symbol
    s_name = get_stock_name(s_symbol)
    s_start_date = config["daily"]["start_date"]
    daily_dir = os.path.join(daily_dir, s_start_date)
    daily_path = os.path.join(daily_dir, s_symbol + s_name[:-1] + ".csv")
    s_data = pd.read_csv(daily_path, encoding="gbk")
    draw_stock_graph(s_data, start_date=graph_start_date, graph=graph_type)
    plt.show()


# 绘制股票走势图
def draw_stock_graph(data,start_date="20200101", delta=35, graph="k"):
    start_date = pd.to_datetime(start_date, format="%Y%m%d")
    data["日期"] = data["日期"].str.replace("/", "-")
    data["日期"] = pd.to_datetime(data["日期"], format="%Y-%m-%d")
    data = data[data["日期"] >= start_date]
    data["日期"] = data["日期"].apply(lambda x: date2num(x))

    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.xlabel("Date")
    plt.ylabel("Price")
    if graph == "k":
        plt.title("K-Graph")
        draw_k_graph(data)
    elif graph == "line":
        plt.title("Line-Graph")
        draw_line_graph(data)
    elif graph == "kline":
        plt.title("K-Line-Graph")
        draw_k_graph(data)
        draw_line_graph(data)
    # 这里加图像设计

    plt.grid(True)
    plt.xlim(num2date(data["日期"].min()), num2date(data["日期"].max()))

    # 设置滑动条
    slider_ax = plt.axes([0.1, 0.02, 0.8, 0.03])
    slider = Slider(slider_ax, 'StartDate', data["日期"].min(), data["日期"].max(), valinit=data["日期"].min())

    def update(val):
        start_date = num2date(slider.val)  # 获取滑块的值，并转换为日期格式
        start_date = date2num(start_date)  
        end_date = num2date(slider.val) + pd.Timedelta(days=delta)  # 获取数据最大日期
        end_date = date2num(end_date)
        filtered_data = data[(data["日期"] >= start_date) & (data["日期"] <= end_date)]  # 筛选大于等于滑块日期的数据
        fig.axes[0].set_xlim(num2date(start_date), num2date(end_date))  # 更新图形显示范围
        fig.axes[0].set_ylim(filtered_data["最低"].min(), filtered_data["最高"].max())  # 更新图形显示范围
        fig.canvas.draw_idle()  # 重新绘制图形

    slider.on_changed(update)  # 绑定update函数到滑块上

    plt.show()

# 绘制K线图
def draw_k_graph(data):
    condition = data["收盘"] > data["开盘"]
    colors = np.where(condition, 'r', 'g')
    plt.bar(data["日期"], data["最高"] - data["最低"], bottom=data["最低"], width=0.1, color=colors, edgecolor='k', align='center')
    plt.bar(data["日期"], data["收盘"] - data["开盘"], bottom=data["开盘"], width=0.6, color=colors, edgecolor='k', align='center')

# 绘制折线图
def draw_line_graph(data):
    plt.plot(data["日期"], data["收盘"], label="收盘价")

# 日期转换int函数
def date2num(date):
    return mdates.date2num(date)

# int转换日期函数
def num2date(num):
    return mdates.num2date(num)

if __name__ == "__main__":
    print("running visulization/utils.py")

    vis_daily()
