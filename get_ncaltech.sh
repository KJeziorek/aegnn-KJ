#! /usr/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1d1EzZzGa4pRRdwOzcnlXCk1rtbyDtWmx" -O annotations.zip && rm -rf /tmp/cookies.txt
unzip annotations.zip
rm annotations.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hr28hw9i9xOR_-KqdTB5aSbj2XkOdZVG" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm data.zip

python3 scripts/split_ncaltech.py

rm -r Caltech101
rm -r Caltech101_annotations

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python3 preprocessing.py --dataset ncaltech101 --num-workers 8
