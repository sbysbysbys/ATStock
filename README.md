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
* 修改ATStock\datasets\akshare.yaml中的root_dir为项目地址
* 修改ATStock\datasets\akshare.yaml中的update_time为最后一次更新时间
* 运行代码
```
python .\datasets\update.py
```
