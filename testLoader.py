import numpy as np
import glob
from PIL import Image

time_steps = 200
new_dimensions = 100
UCSD_FOLDER = 'UCSD_Anomaly_Dataset.v1p2'
train_folders = sorted(glob.glob(UCSD_FOLDER+ '/UCSDped1/Train/*'))
train_images = np.zeros((int(time_steps/3) +1,len(train_folders),new_dimensions,100))

for idx, filename in enumerate(train_folders):
    #print(idx, filename)
    train_files = sorted(glob.glob(filename +"/*"))
    time_step_index = 0
    for index,filename2 in enumerate(train_files):
        if index%3 == 0:
            im = Image.open(filename2)
            #print(filename2)
            im = im.resize((100, new_dimensions))
            train_images[time_step_index,idx,:,:] = np.array(im, dtype=np.float32)/255.0
            time_step_index += 1
print(train_images.shape)
#
np.save(UCSD_FOLDER+ '/UCSD_Anomaly_Dataset.v1p2.npy',train_images)




test_folders = sorted(glob.glob(UCSD_FOLDER+ '/UCSDped1/Test/*/'))
for idx, filename in enumerate(test_folders):
    #print(idx, filename)
    if "_gt" in filename or "Test017" in filename:
        test_folders.pop(idx)
        #print(filename)
#now we take the anomalous frames same way like the MATLAB code from UCSDped1.m 

labels = np.zeros((time_steps,len(test_folders)))

labels[59:152, 0] = 1
labels[49:175, 1] = 1
labels[90:, 2] = 1
labels[30:168, 3] = 1
labels[4:90, 4] = 1
labels[139:, 4] = 1
labels[0:100, 5] = 1
labels[109:, 5] = 1
labels[0:175, 6] = 1
labels[0:94, 7] = 1
labels[0:48, 8] = 1
labels[0:140, 9] = 1
labels[69:165, 10] = 1
labels[129:, 11] = 1
labels[0:156, 12] = 1
labels[0:, 13] = 1
labels[137:, 14] = 1
labels[122:, 15] = 1
labels[53:120, 16] = 1
labels[63:138, 17] = 1
labels[44:175, 18] = 1
labels[30:, 19] = 1
labels[15:107, 20] = 1
labels[7:165, 21] = 1
labels[49:171, 22] = 1
labels[39:135, 23] = 1
labels[76:144, 24] = 1
labels[9:122, 25] = 1
labels[104:, 26] = 1
labels[0:15, 27] = 1
labels[44:113, 27] = 1
labels[174:, 28] = 1
labels[0:180, 29] = 1
labels[0:52, 30] = 1
labels[64:115, 30] = 1
labels[4:165, 31] = 1
labels[0:121, 32] = 1
labels[86:, 33] = 1
labels[14:108, 34] = 1

test_images = np.zeros((int(time_steps/3) +1,len(test_folders),new_dimensions,100))
new_labels = np.zeros((int(time_steps/3) +1,len(test_folders)))

for idx, filename in enumerate(test_folders):
    time_step_index = 0
    test_files = sorted(glob.glob(filename +"/*"))
    for index,filename2 in enumerate(test_files):
        if index%3 == 0:
            im = Image.open(filename2)
            #print(filename2)
            im = im.resize((100, new_dimensions))
            test_images[time_step_index,idx,:,:] = np.array(im, dtype=np.float32)/255.0
            if labels[index][idx] == 1:
               new_labels[time_step_index][idx] = 1 
            time_step_index += 1
print(test_images.shape)
np.save(UCSD_FOLDER+ '/UCSD_Anomaly_Dataset_TestSet.npy',test_images)
'''
labels[59:152, 0] = 1
labels[49:175, 1] = 1
labels[90:, 2] = 1
labels[30:168, 3] = 1
labels[4:90, 4] = 1
labels[139:, 4] = 1
labels[0:100, 5] = 1
labels[109:, 5] = 1
labels[0:175, 6] = 1
labels[0:94, 7] = 1
labels[0:48, 8] = 1
labels[0:140, 9] = 1
labels[69:165, 10] = 1
labels[129:, 11] = 1
labels[0:156, 12] = 1
labels[0:, 13] = 1
labels[137:, 14] = 1
labels[122:, 15] = 1
labels[53:120, 16] = 1
labels[63:138, 17] = 1
labels[44:175, 18] = 1
labels[30:, 19] = 1
labels[15:107, 20] = 1
labels[7:165, 21] = 1
labels[49:171, 22] = 1
labels[39:135, 23] = 1
labels[76:144, 24] = 1
labels[9:122, 25] = 1
labels[104:, 26] = 1
labels[0:15, 27] = 1
labels[44:113, 27] = 1
labels[174:, 28] = 1
labels[0:180, 29] = 1
labels[0:52, 30] = 1
labels[64:115, 30] = 1
labels[4:165, 31] = 1
labels[0:121, 32] = 1
labels[86:, 33] = 1
labels[14:108, 34] = 1
'''
np.save(UCSD_FOLDER+ '/UCSD_Anomaly_labels.npy',new_labels)
'''
    train_files = sorted(glob.glob(filename +"/*"))
    for index,filename2 in enumerate(train_files):
        im = Image.open(filename2)
        print(filename2)
        im = im.resize((32, 32))
        train_images[index,idx,:,:] = np.array(im, dtype=np.float32)/255.0'''
