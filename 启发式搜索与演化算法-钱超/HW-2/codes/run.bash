#!/bin/bash
python main.py --graph-type=gset --gset-id=1 > out1.txt 2>&1 &
python main.py --graph-type=gset --gset-id=2 > out2.txt 2>&1 &
python main.py --graph-type=gset --gset-id=3 > out3.txt 2>&1 &
python main.py --graph-type=gset --gset-id=4 > out4.txt 2>&1 &
python main.py --graph-type=regular > out5.txt 2>&1 &
python main.py --graph-type=ER > out6.txt 2>&1 &
python main.py --graph-type=WS > out7.txt 2>&1 &
python main.py --graph-type=BA > out8.txt 2>&1 &