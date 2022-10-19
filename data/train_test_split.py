# Currently for seabird1
from os import listdir, path
from sklearn.model_selection import train_test_split

image_dir = '/baltic_seabird/seabird6/images/'
image_names = listdir(image_dir)


print('Total number of images: ', len(image_names))


train_idx, val_idx, _, _ = train_test_split(image_names, image_names, test_size = 0.25, random_state = 42)

print('Number of train, val, and test imags: ', len(train_idx), len(val_idx))

print('\n\n')
for name in train_idx:
    print(path.join(image_dir, name))


print('\n\n')
for name in val_idx:
    print(path.join(image_dir, name))

