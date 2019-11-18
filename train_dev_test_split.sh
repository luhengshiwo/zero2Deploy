#/bin/bash
sort -R data/cut.txt > data/sort.txt
# 8089
head -n 6144 data/sort.txt > data/train.txt
tail -n 1536 data/sort.txt > data/tmp.txt
head -n 768 data/tmp.txt > data/dev.txt
tail -n 768 data/tmp.txt > data/test.txt
rm data/tmp.txt
rm data/sort.txt
