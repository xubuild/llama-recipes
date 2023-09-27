# -*- coding: UTF-8 -*-

import os, sys
import pandas as pd
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>")
import json


def yield_file_lines(fip):
    with open(fip) as fin:
        for l in fin:
            l = l.strip()
            if l:
                yield l


def merge(fop, fl):
    all_ll = []
    for f in fl:
        logger.info(f"start {f}")
        ll = json.load(open(f))
        all_ll.extend(ll)
    logger.info(f"total {len(all_ll)}")
    with open(fop, 'w') as fout:
        fout.write("[\n")
        for i, x in enumerate(all_ll):
            if i == len(all_ll) - 1:
                fout.write(f"{json.dumps(x, ensure_ascii=False)}\n")
            else:
                fout.write(f"{json.dumps(x, ensure_ascii=False)},\n")
        fout.write("]\n")


if __name__ == '__main__':
    logger.info(sys.argv)
    fl = sys.argv[2:]
    merge(sys.argv[1], fl)
