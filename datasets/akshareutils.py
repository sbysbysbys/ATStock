import akshare as ak
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta

config_path = ".\\datasets\\akshare.yaml"
# 遍历日期函数
def traverse_dates(start_date, end_date, delta=1):
    start_date = datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.strptime(end_date, '%Y%m%d')
    # print(start_date)
    # print(end_date)
    current_date = start_date
    while current_date <= end_date:
        yield current_date.strftime('%Y%m%d')
        current_date += timedelta(days=delta)

# 获取下一天的日期
def next_date(date):
    date = datetime.strptime(date, '%Y%m%d')
    next_date = date + timedelta(days=1)
    return next_date.strftime('%Y%m%d')

# 比较两个日期的大小,是否是前面的日期大于等于后面的日期
def compare_dates(date_str1, date_str2, date_format='%Y%m%d'):
    date_obj1 = datetime.strptime(date_str1, date_format)
    date_obj2 = datetime.strptime(date_str2, date_format)
    return date_obj1 >= date_obj2

# 获取一只股票的日线数据
def get_one_stock_daily(symbol=0, update=False):
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    s_symbol = symbol
    if(symbol == 0):
        s_symbol = config["one_stock_daily"]["symbol"]
    s_name = get_stock_name(s_symbol)
    s_period = config["one_stock_daily"]["period"]
    s_start_date = config["one_stock_daily"]["start_date"]
    s_end_date = config["one_stock_daily"]["end_date"]
    s_adjust = config["one_stock_daily"]["adjust"]
    save_dir = os.path.join(config["one_stock_daily"]["save_dir"], s_start_date)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    s_header = True
    if update != False:
        update_time = config["update_time"]
        s_start_date = next_date(update_time)
        s_end_date = datetime.now().strftime('%Y%m%d')
        s_header = False

    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol=s_symbol, period=s_period, start_date=s_start_date, end_date=s_end_date, adjust=s_adjust)
    with open(os.path.join(save_dir, s_symbol + s_name[:-1] + ".csv"), 'a', newline='') as f:
        stock_zh_a_hist_df.to_csv(f, index=False, header=s_header)
        f.close()
        
    
# 获取一只股票的分钟线数据
def get_one_stock_one_minute(symbol = 0, update = False):
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    s_symbol = symbol
    if(symbol == 0):
        s_symbol = config["one_stock_one_minute"]["symbol"]
    s_name = get_stock_name(s_symbol)
    s_period = config["one_stock_one_minute"]["period"]
    s_start_date = config["one_stock_one_minute"]["start_date"]
    s_end_date = config["one_stock_one_minute"]["end_date"]
    s_adjust = config["one_stock_one_minute"]["adjust"]
    save_dir = os.path.join(config["one_stock_one_minute"]["save_dir"], s_symbol + s_name[:-1])
    log_file = config["log"]
    if update != False:
        update_time = config["update_time"]
        s_start_date = next_date(update_time)
        s_end_date = datetime.now().strftime('%Y%m%d')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = " 09:30:00"
    end_time = " 15:00:00"
    for date in traverse_dates(s_start_date, s_end_date):
        try:
            stock_zh_a_minute_df = ak.stock_zh_a_hist_min_em(symbol=s_symbol, start_date=date+start_time, end_date=date+end_time, period=s_period, adjust=s_adjust)
            if stock_zh_a_minute_df.empty:
                continue
            save_path = os.path.join(save_dir, date + ".csv")
            with open(save_path, 'w', newline='') as f:
                stock_zh_a_minute_df.to_csv(f, index=False)
                f.close()
        except:
            print("    ", date, "error")
            with open(log_file, 'a', newline='') as f:
                f.write(s_symbol + s_name[:-1] + " " + date + " error\n")
                f.close()
            continue

# 获取所有股票的日线数据
def get_all_stock_daily(update=False):

    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    stock_name_dir = config["stock_name_dir"]
    with open(stock_name_dir, 'r') as f:
        for line in f.readlines():
            line = line.split(",")
            code = line[0]
            print(code)
            get_one_stock_daily(code, update=update)

# 获取所有股票的分钟线数据
def get_all_stock_one_minute(update=False):
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    stock_name_dir = config["stock_name_dir"]
    with open(stock_name_dir, 'r') as f:
        restart = False
        for line in f.readlines():
            line = line.split(",")
            code = line[0]
            if code[0:6] == "002270":
                restart = True
            print(code, restart)
            if restart == False:
                continue
            get_one_stock_one_minute(code, update=update)

# 根据股票代码获取股票名称
def get_stock_name(code):
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    stock_name_dir = config["stock_name_dir"]
    with open(stock_name_dir, 'r') as f:
        for line in f.readlines():
            line = line.split(",")
            if code == line[0]:
                name = line[1]
                name = clean_stock_name(name)
                return name
        f.close()

# 更新股票数据
def update_stock():
    with open(config_path) as f:
        config = yaml.unsafe_load(f)
    update_time = config["update_time"]
    now = datetime.now()
    if now.hour < 18:
        now = now - timedelta(days=1)
    now = now.strftime('%Y%m%d')
    if(compare_dates(update_time, now) == True):
        print("already update")
        return
    get_all_stock_daily(update=True)
    get_all_stock_one_minute(update=True)
    # 更新过后更新更新时间
    now = datetime.now()
    if now.weekday() == 4:
        now = now + timedelta(days=1)
    if now.weekday() == 5:
        now = now + timedelta(days=1)
    now = now.strftime('%Y%m%d')
    print("update time = ", now)
    config["update_time"] = now
    with open(config_path, 'w') as f:
        yaml.dump(config, f)


# 让选择的内容中只出现中文英文或者是数字
def clean_stock_name(name):
    for character in name:
        if not (character.isalpha() or character.isdigit() or character == '\n'):
            name = name.replace(character, '')
    return name

# 清空文件夹下面的空文件
def clean_empty_csvfile(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not file.endswith(".csv"):
                continue
            file_name = os.path.join(root, file)
            if os.path.getsize(file_name) < 70:
                print(file_name)
                os.remove(file_name)



if __name__ == "__main__":
    print("running akshareutils.py")
    # for date in traverse_dates("20200201", "20200215"):
    #     print(date)

    # get_all_stock_daily()

    # stock_zh_a_minute_df = ak.stock_zh_a_hist_min_em(symbol="600036", start_date="20230613 09:30:00", end_date="20230613 15:00:00", period='5', adjust='')
    # print(stock_zh_a_minute_df)

    # get_all_stock_one_minute()

    # get_one_stock_one_minute("688671")

    # update_stock()

    # clean_empty_csvfile("C:\\Users\\admin\\Desktop\\github\\datasets\\akshare\\5_minutes")
    