# coding:utf-8

import urllib.parse
import pandas as pd
import csv


data = csv.reader(open('./data/xss.csv', 'r', encoding="utf-8"))
out = open('./data2/xss.csv', 'a', newline='',encoding='utf-8')
csv_writer = csv.writer(out, dialect='excel')

for each_line in data:
    csv_writer.writerow([urllib.parse.unquote(each_line[0])])
    # print(urllib.parse.unquote(each_line[0]))



