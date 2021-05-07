from base.base_preprocessor import BasePreprocessor
import os
import torch, torchaudio
import librosa as lb
import soundfile as sf
import numpy as np

# Adapted from https://www.kaggle.com/kneroma/clean-fast-simple-bird-identifier-inference
class MelSpecComputer (BasePreprocessor):
    def __init__(self, n_mels, n_fft, f_min, f_max, time_slice, resample_rate, raw_data_dir, processed_data_dir, bulk_process, split_files, active, extensions=None, **kwargs):
        self.sr = resample_rate
        self.n_mels = n_mels
        self.fmin = f_min
        self.fmax = f_max
        self.time_slice = time_slice
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.resample_rate = resample_rate
        self.bulk_process = bulk_process
        self.split_files = split_files
        self.active = active
        #self.n_fft = n_fft
        self.n_fft = self.sr//10
        self.hop_length = self.sr//(10*4)
        #kwargs["n_fft"] = kwargs.get("n_fft", self.n_fft)
        #self.kwargs = kwargs
        overwrite_files = True

        # if bulk_process is true, will process all files in raw_data_dir on init. 
        # See BaseProcessor for implementation details
        super().__init__(raw_data_dir, processed_data_dir, overwrite_files, bulk_process, extensions)

    def process(self, fp, split_files=None):
        audio, orig_sr = sf.read(fp, dtype="float32")
        if self.resample_rate != orig_sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
        if self.split_files:
            self.step = 5 * self.sr
            audios = [audio[i:i+self.time_slice*self.sr] for i in range(0, max(1, len(audio) - self.step + 1), self.step)]
            audios[-1] = self.crop_or_pad(audios[-1], self.time_slice*self.sr)
            images = [self.make_melspec(audio) for audio in audios]
            images = np.stack(images)
            return images
        else:
            melspec = self.make_melspec(audio)
            return melspec  

    def make_melspec(self, y):
        melspec = lb.feature.melspectrogram(
            y, sr=self.sr, n_mels=self.n_mels, n_fft = self.n_fft, fmin=self.fmin, fmax=self.fmax, hop_length=self.hop_length
        )
        melspec = lb.power_to_db(melspec).astype(np.float32)
        melspec = self.process_melspec(melspec)
        return melspec

    def process_melspec(self, melspec):
        melspec = self.mono_to_color(melspec)
        return melspec

    def mono_to_color(self, X, eps=1e-6, mean=None, std=None):
        mean = mean or X.mean()
        std = std or X.std()
        X = (X - mean) / (std + eps)
        
        _min, _max = X.min(), X.max()

        if (_max - _min) > eps:
            V = np.clip(X, _min, _max)
            V = 255 * (V - _min) / (_max - _min)
            V = V.astype(np.uint8)
        else:
            V = np.zeros_like(X, dtype=np.uint8)

        return V

    def crop_or_pad(self, y, length, is_train=False, start=None):
        if len(y) < length:
            y = np.concatenate([y, np.zeros(length - len(y))])
            
            n_repeats = length // len(y)
            epsilon = length % len(y)
            
            y = np.concatenate([y]*n_repeats + [y[:epsilon]])
        
        elif len(y) > length:
            if not is_train:
                start = start or 0
            else:
                start = start or np.random.randint(len(y) - length)

            y = y[start:start + length]

        return y

    # def bulk_processor(self, data_dir, extensions):
    #     if self.active:
    #         super().folder_crawl(data_dir, extensions)
    #     return self.processed_data_dir
        

class MelSpecMaker (BasePreprocessor):
    def __init__(self, n_mels, n_fft, f_min, f_max, time_slice, resample_rate, processed_data_dir, bulk_process, active):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.time_slice = time_slice
        self.resample_rate = resample_rate
        self.processed_data_dir = processed_data_dir
        self.active = active
        self.bulk_process = bulk_process
        overwrite_files = False
        super().__init__(processed_data_dir, overwrite_files, bulk_process)

    # all base class file-crawlers repeatedly call the abstract method proces
    # here, in the child class, we implement the per-file processing
    def process(self, fp):
        resample = torchaudio.transforms.Resample(new_freq = self.resample_rate, resampling_method='sinc_interpolation')
        waveform, sample_rate = torchaudio.load(fp)
        if sample_rate != self.resample_rate:
            resample(new_freq = self.resample_rate, resampling_method='sinc_interpolation')
        # reduce to single channel; downsample
        waveform = torch.mean(input = waveform, dim = 0)
        full_specgram = torchaudio.transforms.MelSpectrogram(n_mels = self.n_mels, sample_rate = self.resample_rate, n_fft = self.n_fft, f_min = self.f_min, f_max = self.f_max, norm = 'slaney')(waveform)
        return full_specgram

    def bulk_process(self, data_dir, extensions):
        if self.active:
            super().folder_crawl(data_dir, extensions)
        return self.processed_data_dir

    def is_active(self):
        return self.active



class Simple_Resampler (BasePreprocessor):
    def __init__(self, n_fft, time_slice, resample_rate, processed_data_dir):
        self.n_fft = n_fft
        self.time_slice = time_slice
        self.resample_rate = resample_rate
        self.processed_data_dir = processed_data_dir
        super().__init__(processed_data_dir)

    # all base class file-crawlers repeatedly call the abstract method proces
    # here, in the child class, we implement the per-file processing
    def process(self, fp, data_dir):
        new_fp = fp.replace(data_dir, self.output_dir)
        new_fp = os.path.splitext(new_fp)[0] + ".pickle"
        waveform, sample_rate = torchaudio.load(fp, normalization = True)
        if sample_rate != 32000:
            torchaudio.transforms.Resample(new_freq = 320000, resampling_method='sinc_interpolation')
        return waveform, new_fp

    def resample(self, data_dir):
        super().folder_crawl(data_dir)



# crop = 2 * self.time_slice * self.resample_rate // self.n_fft + 1
# num_chunks = full_specgram.shape[1]/crop
# if full_specgram.shape[1]%crop > 0:
#     num_chunks += 1
# for chunk in num_chunks:

# class AudioToImage:
#     def __init__(self, n_mels=128, n_fft = 1024, fmin=0, fmax=None, duration=7, sr=32000, step=None, res_type="kaiser_fast"):
#         self.resample=True
#         self.sr = sr
#         self.n_mels = n_mels
#         self.fmin = fmin
#         self.fmax = fmax or self.sr//2

#         self.duration = duration
#         self.audio_length = self.duration*self.sr
#         self.step = step or self.audio_length
        
#         self.res_type = res_type
#         self.resample = resample

#         self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin,
#                                                  fmax=self.fmax)
        
#     def audio_to_image(self, audio):
#         melspec = self.mel_spec_computer(audio) 
#         image = mono_to_color(melspec)
# #         image = normalize(image, mean=None, std=None)
#         return image

#     def __call__(self, row, save=True):
# #       max_audio_duration = 10*self.duration
# #       init_audio_length = max_audio_duration*row.sr
        
# #       start = 0 if row.duration <  max_audio_duration else np.random.randint(row.frames - init_audio_length)
    
#       audio, orig_sr = sf.read(row.filepath, dtype="float32")

#       if self.resample and orig_sr != self.sr:
#         audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)
        
#       audios = [audio[i:i+self.audio_length] for i in range(0, max(1, len(audio) - self.audio_length + 1), self.step)]
#       audios[-1] = crop_or_pad(audios[-1] , length=self.audio_length)
#       images = [self.audio_to_image(audio) for audio in audios]
#       images = np.stack(images)
        
#       if save:
#         path = TRAIN_AUDIO_IMAGES_SAVE_ROOT/f"{row.primary_label}/{row.filename}.npy"
#         path.parent.mkdir(exist_ok=True, parents=True)
#         np.save(str(path), images)
#       else:
#         return  row.filename, images

# def get_audios_as_images(df):
#     pool = joblib.Parallel(2)
    
#     converter = AudioToImage(step=int(DURATION*0.666*SR))
#     mapper = joblib.delayed(converter)
#     tasks = [mapper(row) for row in df.itertuples(False)]
    
#     pool(tqdm(tasks))