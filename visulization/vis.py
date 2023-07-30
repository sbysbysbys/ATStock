import argparse
from utils import vis_daily, vis_5minutes

def vis_parser():
    parser = argparse.ArgumentParser(description="Visualize the data")
    parser.add_argument("--graph", type=str, default='0', 
                        help="k: k-graph; line: line-graph; kline: k-line-graph")
    parser.add_argument("--delta", type=str, default="daily", 
                        help="daily: daily graph; 5minutes: 5 minutes graph")
    parser.add_argument("--symbol", type=str, default='0',
                        help="stock symbol")
    parser.add_argument("--start_date", type=str, default='0',
                        help="start date of the daily graph")
    parser.add_argument("--date", type=str, default='0',
                        help="date of the 5 minutes graph")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = vis_parser()
    if opt.delta == "daily":
        vis_daily(symbol=opt.symbol, graph_start_date=opt.start_date, graph_type=opt.graph)
    elif opt.delta == "5minutes":
        vis_5minutes(symbol = opt.symbol, graph_date = opt.date, graph_type = opt.graph)

