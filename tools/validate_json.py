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


def val(fip, fop):
    new_ll = []
    skip = 0
    ll = json.load(open(fip))
    for i, x in enumerate(ll):
        if 'instruction' not in x:
            if 'input' in x:
                x['instruction'] = x['input']
                del x['input']
                new_ll.append(x)
            else:
                skip = skip + 1
        else:
            new_ll.append(x)
    logger.info(f"total: {len(new_ll)}, skip: {skip}")
    with open(fop, 'w') as fout:
        fout.write("[\n")
        total = len(new_ll)
        for i, l in enumerate(new_ll):
            if i != total - 1:
                fout.write(json.dumps(l, ensure_ascii=False) + ',\n')
            else:
                fout.write(json.dumps(l, ensure_ascii=False) + '\n')
        fout.write("]\n")


if __name__ == '__main__':
    Fire(val)
