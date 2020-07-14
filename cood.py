import requests
import re
from bs4 import BeautifulSoup
import uuid
import os

seasons=['summer','autumn','winter']
for season in seasons:
    i = 0
    for page in range(1, 250):
        print(season)
        if season=='spring':
            r=requests.get('https://wear.jp/women-coordinate/?tag_ids=6026&pageno='+str(page))
        elif season=='summer':
            r=requests.get('https://wear.jp/women-coordinate/?tag_ids=9134&pageno='+str(page))
        elif season=='autumn':
            r = requests.get('https://wear.jp/women-coordinate/?tag_ids=51971&pageno='+str(page))
        elif season=='winter':
            r = requests.get('https://wear.jp/women-coordinate/?tag_ids=4317&pageno='+str(page))

        soup = BeautifulSoup(r.text,'lxml')
        imgs = soup.find_all('img', attrs={"data-originalretina": re.compile('^//cdn.wimg.jp/coordinate')})
        for img in (imgs):
            i = i+1
            print('http:' + img["data-originalretina"])
            r = requests.get('http:' + img["data-originalretina"])

            with open(str('data/')+str(season)+'/'+ str(i)+str('.jpeg'),'wb') as file:
                    file.write(r.content)
            if i>7000:
                break