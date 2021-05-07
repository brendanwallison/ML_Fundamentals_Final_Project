import os
import errno
import numpy as np
import pandas as pd
import splitfolders
import pathlib
import torch
import torchaudio
import zipfile
import glob
import fnmatch
from  ast import literal_eval
# old xeno canto functions
import requests
import json
from collections import Counter 
import csv
from torchvision.utils import save_image

class birdclef2:
    def __init__(self, preprocessor = None, mode = "xeno", vanilla = True, download = True, transform = [], data_dir = "data/external", from_file = True, alpha = 1):
        self.split_files = True
        self.from_file = from_file
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        self.vanilla = vanilla
        self.mode = mode
        self.train_meta = pd.read_csv(os.path.join(data_dir, 'train_metadata.csv'))
        self.train_soundscapes = pd.read_csv(os.path.join(data_dir, 'train_soundscape_labels.csv'))
        self.test_data = pd.read_csv(os.path.join(data_dir, 'test.csv') )
        self.transform = transform
        if self.mode == "xeno":
            self.meta = self.train_meta
        elif self.mode == "soundscape":
            self.meta = self.train_soundscapes
            self.group_meta = self.meta.groupby(by='audio_id', sort=False)
        elif self.mode == "test":
            self.meta = self.test_data
        self.labels = {label: label_id for label_id,label in enumerate(sorted(self.train_meta["primary_label"].unique()))}
        self.labels_inv = [label for label in sorted(self.train_meta["primary_label"].unique())]
        self.num_classes = len(self.labels)
        self.alpha = alpha

        if download:
            self.download()
        if preprocessor != None:
            self.horizontal_crop = 2 * preprocessor.time_slice * preprocessor.resample_rate // preprocessor.n_fft + 1
            if preprocessor.bulk_process:
                self.from_file = True  

        if self.from_file:
            #self.meta['new_filename'] = self.meta['filename'] + '.npy'
            self.meta['new_filename'] = [os.path.splitext(filename)[0] + '.pickle' for filename in self.meta['filename']]
            print("Updated filenames")
        # else:
        #     self.horizontal_crop = 256

    def append_folder(self, path):
        if self.mode == "xeno":
            p_dir = os.path.join(path, 'train_short_audio')
        elif self.mode == "soundscape":
            p_dir = os.path.join(path, 'train_soundscapes')
        elif self.mode == "test":
            p_dir = os.path.join(path, 'test_soundscapes')         
        return p_dir
           
    def set_transform(self, trsfm):
        self.transform = [trsfm]

    def __len__(self):
        if self.mode == 'soundscape':
            return len(self.group_meta.size())
        return self.meta.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'soundscape':
            groupname = self.group_meta.first().iloc[idx,:].name
            start_idx = self.group_meta.groups[groupname][0]
            num_images = self.group_meta.size()[groupname]
            label = [self.smooth_labels(start_idx + i) for i in range(num_images)]
            label = np.stack(label)
            idx = start_idx
        else:
            label = self.smooth_labels(idx)

        if self.from_file:
            item = self.get_image_item(idx)
        else:
            item = self.get_audio_item(idx)

        for trsfm in self.transform:
            item = trsfm(item)
        return (item, label)

    # inspired by https://www.kaggle.com/kneroma/clean-fast-simple-bird-identifier-training-colab   
    # soundscape labels verified
    def smooth_labels(self, idx):
        row = self.meta.iloc[idx, :]
        secondaries = []
        if self.mode == 'xeno':
            primaries = [row['primary_label']]
            secondaries = literal_eval(row['secondary_labels'])
        elif self.mode == 'soundscape':
            primaries = row['birds'].split(' ')
        a = self.alpha/self.num_classes
        if self.vanilla:
            t = torch.zeros([self.num_classes]) + a  
            for primary in primaries: 
                if primary == 'nocall':
                    break
                else:
                    t[self.labels.get(primary)] = 1 - a
            for secondary in secondaries:
                secondary_index = self.labels.get(secondary)
                if secondary_index:
                    t[secondary_index] = 0.5*(1 - a)
        else:
            t = t
        return t
    
    def get_image_item(self, idx):
        row = self.meta.loc[idx, :]
        if self.mode == 'xeno':
            primary = row['primary_label']
            dir = self.append_folder(self.data_dir)
            fp = os.path.join(dir, primary, row['new_filename'])
            if self.data_dir == 'data/external':
                item = np.load(fp)
            else:
                item = torch.load(fp)
        # to-do: make soundscape loading more efficient;
        # currently loads the entire relevant file instead of a 5s slice    
        elif self.mode == 'soundscape':
            id = str(row['audio_id'])
            site = str(row['site'])
            expression = id + '_' + site + '*'
            filename = fnmatch.filter(os.listdir(self.data_dir), expression)
            path = os.path.join(self.data_dir, filename[0])
            end_time = row['seconds']
            start_time = end_time - self.time_slice
            if start_time > 0:
                start_pixel = self.sp(start_time)
            else:
                start_pixel = 0
            end_pixel = start_pixel + self.sp(time_slice)
            item = np.load(path)[:, start_pixel:end_pixel]
        return item

    def get_audio_item(self, idx):
        row = self.meta.loc[idx, :]
        if self.mode == 'soundscape':
            id = str(row['audio_id'])
            site = str(row['site'])
            expression = id + '_' + site + '*'
            dir = os.path.join(self.data_dir, 'train_soundscapes')
            filename = fnmatch.filter(os.listdir(dir), expression)
            path = os.path.join(dir, filename[0])
            melspec_stack = self.preprocessor.process(path, self.split_files)
            return melspec_stack
        elif self.mode == 'xeno':
            path = self.xeno_fp(idx)
            melspec_stack = self.preprocessor.process(path, self.split_files)
            return melspec_stack


    def xeno_fp(self, idx):
        row = self.meta.loc[idx, :]
        primary = row['primary_label']
        dir = self.append_folder(self.data_dir)
        fp = os.path.join(dir, primary, row['filename'])
        return fp

    def findKaggle(self):      
        os.chdir('/')
        print("Current directory:", os.getcwd())
        for root, dirs, files in os.walk(os.getcwd()):
            if '.kaggle' in dirs and root != '/root':
                return os.path.join(root, '.kaggle')

    def sp(self, seconds):
        i = 2 * seconds * self.preprocessor.resample_rate // self.preprocessor.n_fft  
        return int(i)

    def finalize_prediction(self, output, target):
        pred = torch.where(output > 0.3, 1, 0)
        label = torch.where(target > 0.5, 1, 0)
        return pred, label

    def prediction_string(self, pred, label):
        predictions, labels = self.finalize_prediction(pred, label)
        assert predictions.shape[0] == labels.shape[0]
        assert predictions.shape[1] == labels.shape[1]
        df = pd.DataFrame(data=np.empty((predictions.shape[0], 2), dtype = np.str))
        for i in range(predictions.shape[0]):
            pred_list = ""
            label_list = ""
            for c in range(predictions.shape[1]):
                if predictions[i, c] == 1:
                    pred_list += self.get_label_from_id(c) + ' '
                if labels[i,c] == 1:
                    label_list += self.get_label_from_id(c) + ' '
            if label_list == '':
                label_list = 'nocall '
            if pred_list == '':
                pred_list = 'nocall '
            pred_list = pred_list[:-1]
            label_list = label_list[:-1]
            df.iloc[i,0] = pred_list
            df.iloc[i,1] = label_list
        return df

    def get_label_from_id(self, id):
        return self.labels_inv[id]

    def get_id_from_label(self, label):
        return self.labels[label]
                
    def download(self):
        owd = os.getcwd()
        try:
            pathlib.Path(self.download_data_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            pass
        download = 'kaggle competitions download -c birdclef-2021 -p ' + os.path.join(owd, self.download_data_dir)
        successful_download = os.system(download)      
        if successful_download != 0:
            folder_path = self.findKaggle()
            cmd = 'cp -r ' + folder_path +  ' /root'
            found_kaggle_folder = os.system(cmd)   
            if found_kaggle_folder == 0:
                os.system(download)    
        expected_fp = os.path.join(self.download_data_dir, 'birdclef-2021.zip')
        if os.path.exists(expected_fp):
            with zipfile.ZipFile(expected_fp, 'r') as zip_ref:
                zip_ref.extractall(self.download_data_dir)



#############################


class birdclef_old:
    def __init__(self, preprocessor = None, mode = "xeno", vanilla = True, download = True, transform = [], download_data_dir = "data/download", alpha = 1):
        self.preprocessor = preprocessor
        self.download_data_dir = download_data_dir
        self.vanilla = vanilla
        self.mode = mode
        self.train_meta = pd.read_csv(os.path.join(download_data_dir, 'train_metadata.csv'))
        self.train_soundscapes = pd.read_csv(os.path.join(download_data_dir, 'train_soundscape_labels.csv'))
        self.test_data = pd.read_csv(os.path.join(download_data_dir, 'test.csv') )
        self.transform = transform
        if self.mode == "xeno":
            self.meta = self.train_meta
        elif self.mode == "soundscape":
            self.meta = self.train_soundscapes
        elif self.mode == "test":
            self.meta = self.test_data
        self.labels = {label: label_id for label_id,label in enumerate(sorted(self.train_meta["primary_label"].unique()))}
        self.labels_inv = [label for label in sorted(self.train_meta["primary_label"].unique())]
        self.labels['nocall'] = len(self.labels)
        self.labels_inv.append('nocall')
        self.num_classes = len(self.labels)
        self.alpha = alpha
        owd = os.getcwd()
        try:
            pathlib.Path(self.download_data_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            pass
        if download:
            download = 'kaggle competitions download -c birdclef-2021 -p ' + os.path.join(owd, self.download_data_dir)
            successful_download = os.system(download)      
            if successful_download != 0:
                folder_path = self.findKaggle()
                cmd = 'cp -r ' + folder_path +  ' /root'
                found_kaggle_folder = os.system(cmd)   
                if found_kaggle_folder == 0:
                    os.system(download)    
            expected_fp = os.path.join(self.download_data_dir, 'birdclef-2021.zip')
            if os.path.exists(expected_fp):
                with zipfile.ZipFile(expected_fp, 'r') as zip_ref:
                    zip_ref.extractall(self.download_data_dir)
        if preprocessor != None:
            self.horizontal_crop = 2 * preprocessor.time_slice * preprocessor.resample_rate // preprocessor.n_fft + 1  
            p_dir = self.append_folder(self.download_data_dir)                
            self.data_dir = preprocessor.make_spectrograms(p_dir, '.ogg')
            self.data_dir = self.append_folder(self.data_dir)
            if self.mode == 'xeno':
                self.meta['new_filename'] = self.meta['filename'].str.replace('.ogg', '.pickle')
            print("Updated filenames")
        else:
            self.data_dir = self.download_data_dir
            self.horizontal_crop = 256

    def append_folder(self, path):
        if self.mode == "xeno":
            p_dir = os.path.join(path, 'train_short_audio')
        elif self.mode == "soundscape":
            p_dir = os.path.join(path, 'train_soundscapes')
        elif self.mode == "test":
            p_dir = os.path.join(path, 'test_soundscapes')         
        return p_dir
           
    def set_transform(self, trsfm):
        self.transform = [trsfm]

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        item = self.getItem(idx)
        label = self.smooth_labels(idx)
        for trsfm in self.transform:
            item = trsfm(item)
        return (item, label)

    # inspired by https://www.kaggle.com/kneroma/clean-fast-simple-bird-identifier-training-colab   
    def smooth_labels(self, idx):
        row = self.meta.iloc[idx, :]
        secondaries = []
        if self.mode == 'xeno':
            primaries = [row['primary_label']]
            secondaries = literal_eval(row['secondary_labels'])
        elif self.mode == 'soundscape':
            primaries = row['birds'].split(' ')
        a = self.alpha/self.num_classes
        if self.vanilla:
            t = torch.zeros([self.num_classes]) + a  
            for primary in primaries: 
                t[self.labels.get(primary)] = 1 - a
            for secondary in secondaries:
                secondary_index = self.labels.get(secondary)
                if secondary_index:
                    t[secondary_index] = 0.5*(1 - a)
        else:
            t = t
        return t
    
    def getItem(self, idx):
        row = self.meta.loc[idx, :]
        if self.mode == 'xeno':
            primary = row['primary_label']
            fp = os.path.join(self.data_dir, primary, row['new_filename'])
            item = torch.load(fp)
        # to-do: make soundscape loading more efficient;
        # currently loads the entire relevant file instead of a 5s slice    
        elif self.mode == 'soundscape':
            id = str(row['audio_id'])
            site = str(row['site'])
            expression = id + '_' + site + '*'
            filename = fnmatch.filter(os.listdir(self.data_dir), expression)
            path = os.path.join(self.data_dir, filename[0])
            end_time = row['seconds']
            start_pixel = self.sp(end_time - 5)
            end_pixel = self.sp(end_time)
            item = torch.load(path)[:, start_pixel:end_pixel]
        return item

    def findKaggle(self):      
        os.chdir('/')
        print("Current directory:", os.getcwd())
        for root, dirs, files in os.walk(os.getcwd()):
            if '.kaggle' in dirs and root != '/root':
                return os.path.join(root, '.kaggle')

    def sp(self, seconds):
        i = 2 * seconds * self.preprocessor.resample_rate // self.preprocessor.n_fft  
        return int(i)

    def finalize_prediction(self, output, target):
        pred = torch.where(output > 0.4, 1, 0)
        label = torch.where(target > 0.5, 1, 0)
        pred = pred[:, :-1]
        label = label[:, :-1]
        return pred, label

    def prediction_string(self, pred, label):
        predictions, labels = self.finalize_prediction(pred, label)
        assert predictions.shape[0] == labels.shape[0]
        assert predictions.shape[1] == labels.shape[1]
        df = pd.DataFrame(data=np.empty((predictions.shape[0], 2), dtype = np.str))
        for i in range(predictions.shape[0]):
            pred_list = ""
            label_list = ""
            for c in range(predictions.shape[0]):
                if predictions[i, c] == 1:
                    pred_list += self.get_label_from_id(c) + ' '
                if labels[i,c] == 1:
                    label_list += self.get_label_from_id(c) + ' '
            if label_list == '':
                label_list = 'nocall'
            if pred_list == '':
                pred_list = 'nocall'
            df.iloc[i,0] = pred_list
            df.iloc[i,1] = label_list
        return df

    def get_label_from_id(self, id):
        return self.labels_inv[id]

    def get_id_from_label(self, label):
        return self.labels[label]


# ###########
#         meta_path = os.path.join(self.input_dir, self.metadata_filename)
#         meta = pd.read_csv(meta_path)
#         downsample = torchaudio.transforms.Resample(new_freq = self.resample_rate, resampling_method='sinc_interpolation')
#         for count, filename in enumerate(meta['filename']):
#             filepath = os.path.join(self.input_dir, filename)
#             new_filename = os.path.splitext(filename)[0] + ".pickle"
#             new_filepath = os.path.join(self.output_dir, new_filename)
#             if not os.path.exists(filepath): 
#                 print(filepath + " could not be found\n")
#                 return False
#             else:
#                 # waveform, sample_rate = torchaudio.load(filename)
#                 # remake the spectrograms even if they already exist
#                 waveform, sample_rate = torchaudio.load(filepath, normalization = True)
#                 waveform = torch.mean(input = waveform, dim = 0) 
#                 waveform = downsample(waveform)

#                 # if audio file is too short, duplicate its spectrogram (if still too short, will be padded later)
#                 mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)  # (channel, n_mels, time)
#                 if self.min_waveform_width > len(waveform):
#                     mel_specgram = torch.cat((mel_specgram, mel_specgram), 1) 
#                 torch.save(mel_specgram, new_filepath)
#                 meta['filename'][count] = new_filepath
#         meta.to_csv(meta_path)

        
# given a location string, find n most common birds by number of recordings in xeno canto database
class XenoCommonBirds:
    def __init__(self, download_data_dir, split_data_dir, location = 'Texas', num_birds = 5, metadata_filename = "metadata.csv", split=True, train_split = .7, test_split = .15, n_fft = 400, time_slice = 10, resample_rate = 22050, num_workers=1):
        self.num_workers = num_workers 

        # download control 
        self.download_data_dir = download_data_dir
        self.location = location
        self.num_birds = num_birds
        self.metadata_filename = metadata_filename   

        # test/train/valid folder control
        self.split_data_dir = split_data_dir
        self.train_split = train_split
        self.test_split = test_split
        self.valid_split = 1 - test_split - train_split 
        self.split = split


    # find n most common birds in the given location
    def download_n_common_birds(self):
        birdcount = 0
        payload = {'query': 'loc:' + self.location, 'page' : 1}
        r = requests.get('https://www.xeno-canto.org/api/2/recordings', params=payload)
        birds = r.json()
        # count 
        c = Counter( (bird['gen'], bird['sp']) for bird in r.json()['recordings']  )
        numPages = birds['numPages']
        for i in range(2, numPages):
            birds = r.json()
            c += Counter( (bird['gen'], bird['sp']) for bird in r.json()['recordings']  )
            i+=1
        
        # initialize metadata array
        # metadata = [['database', 'id', 'length', 'filename', 'audible_birds']]
        metadata = [['database', 'id', 'length', 'filename', 'foreground bird', 'audible_birds']]

        # Work in data/download folder, creating it if necessary
        owd = os.getcwd()
        try:
            pathlib.Path(self.download_data_dir).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            if error.errno != errno.EEXIST:
                raise
            pass
        os.chdir(self.download_data_dir)

        for common_bird in range(self.num_birds):
            # request first page of results
            gen = c.most_common(self.num_birds)[common_bird][0][0]
            sp = c.most_common(self.num_birds)[common_bird][0][1]
            full_name = gen + "_" + sp
            full_name_for_search = gen + " " + sp

            try:
                os.mkdir(full_name.replace(' ', '_'))
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise
                pass
            
            result_page = 1
            # payload = {'query': full_name_for_search, 'loc': self.location, 'page' : i} # restrict to location
            payload = {'query': full_name_for_search, 'page' : result_page}
            r = requests.get('https://www.xeno-canto.org/api/2/recordings', params=payload)
            birds = r.json()
            numPages = birds['numPages']

            # work through numPages pages of results
            while result_page <= numPages and result_page < 3:
                # already made request for i==1
                if result_page > 1:
                    payload = {'query': 'loc:' + self.location, 'page' : result_page, 'gen' : gen, 'sp': sp}
                    r = requests.get('https://www.xeno-canto.org/api/2/recordings', params=payload)
                    birds = r.json()
                for bird in r.json()['recordings']:
                    # build filepath, replacing unwanted characters
                    ext = os.path.splitext(bird['file-name'])[1]
                    rec_id = bird['id']
                    file_path = os.path.join(full_name, rec_id + ext) # n common birds, n corresponding directories
                    # file_path = rec_id + ext # all files in common folder

                    # build metadata
                    if (bird['also']) == list(['']): # only list main bird
                        audible_birds = list([full_name_for_search])
                    else: # list all audible birds (avoid duplicates via sets)
                        audible_birds = list(set(bird['also']).union(set([full_name_for_search])))    
                    
                    # add metadata regardless of download success
                    metadata_row = ['xeno canto', bird['id'], bird['length'], file_path, full_name, audible_birds]
                    metadata.append(metadata_row)
                    birdcount += 1

                    # download file
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        print(file_path + " already exists...proceeding to next file\n")
                    else:
                        f = requests.get('http:' + bird['file'])
                        if f.status_code == 200: # all good with file
                            # binary download of file, chunk by chunk
                            with open(file_path, 'wb') as fd:
                                for chunk in f.iter_content(chunk_size=128):
                                    fd.write(chunk)
                            # # only add metadata for each successfully downloaded file
                            # metadata_row = ['xeno canto', bird['id'], bird['length'], fp, audible_birds]
                            # metadata.append(metadata_row)
                            # birdcount += 1
                result_page+=1

        
        with open(self.metadata_filename, 'w') as f:       
            write = csv.writer(f) 
            write.writerows(metadata) 

        os.chdir(owd)
        print("Number of birds catalogued is: " + str(birdcount))


    # only checks that the files have been created, not that they were downloaded properly
    def files_present(self):
        owd = os.getcwd()
        os.chdir(self.download_data_dir)
        md = pd.read_csv(self.metadata_filename)
        for filename in md['filename']:
            if not os.path.exists(filename): 
                print(filename + " does not exist\n")
                os.chdir(owd)
                return False
        os.chdir(owd)
        return True

    def split_folders(self):
        splitfolders.ratio(self.download_data_dir, ratio=(self.test_split, self.train_split, self.valid_split), output=self.split_data_dir)

