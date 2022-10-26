import pandas as pd
from asari.api import Sonar
# from math import floor
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

sonar = Sonar()

df_wrime = pd.read_table(
    '/Users/yukitakeyama/Documents/yukai-code/wrime/wrime-ver2.tsv')
cnt = 0
max_count = 100000
senti = [0] * 4
correct_rate = [0]*len(df_wrime)

for (text, senti[0], senti[1], senti[2], senti[3]) in zip(df_wrime['Sentence'], df_wrime['Writer_Sentiment'], df_wrime['Reader1_Sentiment'], df_wrime['Reader2_Sentiment'], df_wrime['Reader3_Sentiment']):
    # print(str)
    res = sonar.ping(text)
    correct_senti = Decimal(
        str(sum(senti)/4)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
    if res['top_class'] == 'negative' and correct_senti < 0:
        print('○', end=' ')
        correct_rate[cnt] = 1
    elif res['top_class'] == 'positive' and correct_senti > 0:
        print('○', end=' ')
        correct_rate[cnt] = 1
    else:
        print('❌', end=' ')
        correct_rate[cnt] = 0

    print(res['top_class'], end=' ')
    print(senti, correct_senti, end='\n')

    cnt += 1
    if cnt == max_count or cnt == len(df_wrime):
        print('正解率: ', sum(correct_rate)/cnt)
        break
