# -*- coding: utf-8 -*-
import sys
import torch
from config import PATH, openLog
from data_processing import decode

def main(batch_size=1):
    model = torch.load(PATH+'generator.pkl')
    reverse_vocab = torch.load(PATH+'reverse_vocab.pkl')

    num = model.generate(batch_size=batch_size)
    log = openLog('genTxt_predict.txt')
    result = decode(num, reverse_vocab, log)
    log.close()
    from wordsegment import load, segment
    load()
    lines = []
    with open(PATH + 'genTxt_predict.txt', encoding='latin') as f:
        # f.seek(0,0)
        lines = f.readlines()
        print(lines, "abcd")
    open('generated.txt', 'w').close()
    for line in lines:

        temp = segment(line)
        str1 = ""
        for j in range(len(temp)):
            str1 += temp[j] + " "
        with open("generated.txt", "a") as file:
            file.write(str1 + "\n")
    return result

if __name__ == '__main__':
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 1
    
    result = main(batch_size)
    print(result)
    