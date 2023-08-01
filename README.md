# ATStock
## 下载
* 下载仓库
```
git clone https://github.com/sbysbysbys/ATStock.git
```
* 安装库
* 下载anaconda3，建立虚拟环境
```
conda create -n ATStock python=3.9
conda activate ATStock
```
* 安装库
```
pip install -r requirements.txt
```

## 日常更新
* 修改ATStock\datasets\akshare.yaml中的root_dir为akshare数据下载地址
* 修改ATStock\datasets\akshare.yaml中的update_time为最后一次更新时间
* 运行代码
```
python .\datasets\update.py
```

## 可视化
* 修改ATStock\visulization\vis.yaml中的root_dir为所有数据保存地址
* 参考ATStock\visulization\utils.py中的draw_stock_graph()编写自己的图像
* 运行代码，参数可以在下面进行设置，也可以在vis.yaml中进行设置，下面的参数设置优先
```
# 日线
python .\visulization\vis.py --delta daily [--graph kline] [--symbol 000001] [--start_date 20220101]
# 五分钟线
python .\visulization\vis.py --delta 5minutes [--graph kline] [--symbol 000001] [--date 20230725]
```
