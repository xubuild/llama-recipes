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


def conv_one(dd):
    new_item = [
        {
            "role": "system",
            "content": f"you are a professional advisor on sports, training and health, and you should provide accurate and none-harmful answers to user's question. " +
                       f"Do not repeat or translate the question, and only respond with the answer. " +
                       f"if you are not sure about the question's meaning, you can reply that you don't know. " +
                       f"your answer must be in Chinese."
        },
        {"role": "user", "content": dd['instruction']}
    ]
    return new_item


def conv(fip, fop):
    ll = []
    for l in yield_file_lines(fip):
        try:
            dd = json.loads(l)
            ll.append(conv_one(dd))
        except Exception as e:
            logger.error(f'{l} {e}')

    with open(fop, 'w') as fout:
        fout.write("[\n")
        totla = len(ll)
        for i, l in enumerate(ll):
            if i != totla - 1:
                fout.write(json.dumps(l, ensure_ascii=False) + ',\n')
            else:
                fout.write(json.dumps(l, ensure_ascii=False) + '\n')
        fout.write("]\n")


if __name__ == '__main__':
    conv(sys.argv[1], sys.argv[2])
