import csv
from data import *
import pandas as pd

out = open('test2.csv','a', newline='')
csv_write = csv.writer(out,dialect='excel')

train_data_sql = read_data('./1.txt','0')

csv_write.writerow(['content', 'type'])

for data_list in train_data_sql:
    csv_write.writerow(data_list)


print('ok!')

print(pd.read_csv('./test2.csv').head(2))