import os
import shutil

directory = 'Caltech101'
ann_directory = 'Caltech101_annotations'

dst_train = 'data/storage/ncaltech101/training'
dst_test = 'data/storage/ncaltech101/test'
dst_val = 'data/storage/ncaltech101/validation'

train = 0.8
test = 0.1

for subdir, dirs, files in os.walk(directory):
    for dire in dirs:
        print(dire)
        if dire == 'BACKGROUND_Google':
            continue
        os.makedirs(os.path.join(dst_train, dire), exist_ok=True)
        os.makedirs(os.path.join(dst_test, dire), exist_ok=True)
        os.makedirs(os.path.join(dst_val, dire), exist_ok=True)

        size = len(os.listdir(os.path.join(directory, dire)))
        train_id = int(size*train)
        test_id = int(size*(train+test))
        for id, filename in enumerate(os.listdir(os.path.join(directory, dire))):
            src = os.path.join(directory, dire, filename)
            annot = os.path.join(ann_directory, dire, filename.replace("image", "annotation"))
            
            if id < train_id:
                shutil.copyfile(src, os.path.join(dst_train, dire, filename))
                shutil.copyfile(annot, os.path.join(dst_train, dire, filename.replace("image", "annotation")))
            elif id < test_id:
                shutil.copyfile(src, os.path.join(dst_test, dire, filename))
                shutil.copyfile(annot, os.path.join(dst_test, dire, filename.replace("image", "annotation")))
            else:
                shutil.copyfile(src, os.path.join(dst_val, dire, filename))
                shutil.copyfile(annot, os.path.join(dst_val, dire, filename.replace("image", "annotation")))