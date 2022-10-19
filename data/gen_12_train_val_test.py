from os import listdir, path
from sklearn.model_selection import train_test_split


train_txt = '/baltic_seabird/train/all.txt'
test_txt = '/baltic_seabird/test/all.txt'
data_dir = '../../data/'

all_txt = list()
# Reading names of all image files
with open(train_txt) as txt:
    img_list = txt.readlines()
    for img in img_list:
        all_txt.append(img)

with open(test_txt) as txt:
    img_list = txt.readlines()
    for img in img_list:
        all_txt.append(img)


print('Total number of images: ', len(all_txt))
idx_list = [i for i in range(len(all_txt))]


# Creating 13 random splits sets
for rs in range(1, 13):
    train_idx, test_idx, _, _ = train_test_split(idx_list, idx_list, test_size = 0.2, random_state = rs)
    train_idx, val_idx, _, _ = train_test_split(train_idx, train_idx, test_size = 0.1, random_state = rs)
    
    
    print('Number of train, val, and test imags: ', len(train_idx), len(val_idx), len(test_idx))

    # Creating txt file
    with open(f'train/train{rs}.txt', 'w') as f:
        for i in train_idx:
            f.write(all_txt[i])

    with open(f'test/test{rs}.txt', 'w') as f:
        for i in test_idx:
            f.write(all_txt[i])
    
    with open(f'val/val{rs}.txt', 'w') as f:
        for i in val_idx:
            f.write(all_txt[i])