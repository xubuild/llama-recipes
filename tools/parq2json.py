# -*- coding: UTF-8 -*-

import os, sys
import pandas as pd
from loguru import logger
from fire import Fire
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>")
import json


def yield_file_lines(fip):
    with open(fip) as fin:
        for l in fin:
            l = l.strip()
            if l:
                yield l

import pyarrow.parquet as pq
import pandas as pd



def conv(fip,fop):


    table = pq.read_table(fip)
    data = table.to_pandas().to_dict(orient='records')

    logger.info(f"total {len(data)}")
    with open(fop, 'w') as fout:
        fout.write("[\n")
        for i, x in enumerate(data):
            if i == len(data) - 1:
                fout.write(f"{json.dumps(x, ensure_ascii=False)}\n")
            else:
                fout.write(f"{json.dumps(x, ensure_ascii=False)},\n")
        fout.write("]\n")


if __name__ == '__main__':

    Fire(conv)
