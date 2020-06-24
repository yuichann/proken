import csv
import os
with open('new_csv.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['img_name','season'])

filename = '../fashiondata_tng/photos'
par_id = os.listdir(filename)
season=None
for count in par_id:
    dir_label = os.path.join(filename, count)
    for file_name in os.listdir(dir_label):
        img_name = os.path.join(dir_label,file_name)
        #print(img_name)
        if img_name.find('spring') != -1:
            season= 0
        elif img_name.find('summer') != -1:
            season= 1
        elif img_name.find('fall') != -1:
            season = 2
        elif img_name.find('autumn') != -1:
            season = 2
        elif img_name.find('winter') != -1:
            season = 3
        else: continue

        with open('new_csv.csv', 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([img_name, season])