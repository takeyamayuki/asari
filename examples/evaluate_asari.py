import pandas as pd
from asari.api import Sonar

sonar = Sonar()

df_wrime = pd.read_table(
    '/Users/yukitakeyama/Documents/yukai/wrime_test/wrime-ver1.tsv')
# print(df_wrime)
cnt = 0

for str in df_wrime['Sentence']:
    # print(str)
    print(sonar.ping(str), end='\n')
    cnt += 1
    if cnt == 10:
        break
