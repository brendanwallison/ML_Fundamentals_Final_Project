import os
import torch, torchaudio
import itertools
from shutil import copyfile


class BasePreprocessor:
    def __init__(self, data_dir, output_dir, overwrite_files, bulk_process, extensions = None):
        # organization
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.overwrite_files = overwrite_files
        self.extensions = extensions or ['.ogg']
        if(bulk_process):
            self.bulk_processor(data_dir, extensions)

    def folder_crawl(self, data_dir, extensions):
        count = 0
        for root, dirs, files in os.walk(data_dir):
            count += 1
            for name in files:
                fp = os.path.join(root, name)
                new_root = root.replace('download', 'processed')
                new_fp = os.path.join(new_root, name)
                new_fp, extension = os.path.splitext(new_fp)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                if (extension in extensions): # process these files
                    new_fp = new_fp + ".pickle"
                    if self.overwrite_files or not os.path.exists(new_fp): # do not overwrite existing processed files
                        print("Processing: ", fp, " to ", new_fp, " file ", count)
                        new_obj = self.process(fp)
                        torch.save(new_obj, new_fp)
                    else:
                        print("Skipping: ", fp, " to ", new_fp, " file ", count)
                else: # copy these files
                    print("Copying: ", fp, " to ", new_fp, " file ", count)
                    copyfile(fp, new_fp + extension)

    def bulk_processor(self, data_dir = '/', extensions = '.ogg'):
        self.folder_crawl(data_dir, extensions)
        return self.processed_data_dir
        
    # default processes; override
    def process(self, fp):
        waveform, sample_rate = torchaudio.load(fp, normalization = True)
        if sample_rate != 32000:
            torchaudio.transforms.Resample(new_freq = 320000, resampling_method='sinc_interpolation')
        return waveform