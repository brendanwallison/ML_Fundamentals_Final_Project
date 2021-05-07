import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import data_loader.datasets as module_dataset
import data_loader.preprocessors as module_preprocessor
import librosa
import matplotlib as plt
from torchvision.utils import save_image
import os
import glob
import pandas as pd

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
  fig, axs = plt.subplots(1, 1)
  axs.set_title(title or 'Spectrogram (db)')
  axs.set_ylabel(ylabel)
  axs.set_xlabel('frame')
  im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
  if xmax:
    axs.set_xlim((0, xmax))
  fig.colorbar(im, ax=axs)
  plt.show(block=False)



def main(config):
    logger = config.get_logger('test')
    preprocessor = config.init_obj('preprocessor', module_preprocessor)
    test_dataset = config.init_obj('dataset', module_dataset, preprocessor, mode = 'xeno', vanilla = True)
    data_loader = config.init_obj('data_loader', module_data, test_dataset)

    # for key, value in test_dataset.labels.items():
    #     if value == 164:
    #         print(key)

    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)
    # state_dict = checkpoint['state_dict']
    # if config['n_gpu'] > 1:
    #     model = torch.nn.DataParallel(model)
    # model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    #model(test_dataset.__getitem__(240)[0][None,:,:,:].to(device))

    dir = "image_samples"
    df = pd.DataFrame(columns=['filename', 'target', 'prediction'])
    if not os.path.exists(dir):
        os.mkdir(dir)
    else:
        files = glob.glob(os.path.join(dir, '*'))
        for f in files:
            os.remove(f)
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            #output = model(data)
            for j in range(20):
                fn = 'new_img' + str(i) + '-' + str(j) + '.png'
                save_image(data[j], os.path.join(dir, fn))

            #df = test_dataset.prediction_string(output, target)

            # computing loss, metrics on test set
            #loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)



# import pandas as pd
# import os

# path = 'data/processed/'
# train_labels = pd.read_csv(path+'train_soundscape_labels.csv')
# train_meta = pd.read_csv(path+'train_metadata.csv')
# test_data = pd.read_csv(path+'test.csv')

# labels = {label: label_id for label_id,label in enumerate(sorted(train_meta["primary_label"].unique()))}
# print(labels['yetvir'])
# print(labels['nocall'])
# labels['nocall'] = len(labels)
# print(labels['nocall'])


# labels = []
# for row in train_labels.index:
#     labels.extend(train_labels.loc[row, 'birds'].split(' '))
# labels = list(set(labels))

# df_labels_train = pd.DataFrame(index=train_labels.index, columns=labels)
# for row in train_labels.index:
#     birds = train_labels.loc[row, 'birds'].split(' ')
#     for bird in birds:
#         df_labels_train.loc[row, bird] = 1
# df_labels_train.fillna(0, inplace=True)

# # We set a dummy value for the target label in the test data because we will need for the Data Generator
# test_data['birds'] = 'nocall'

# df_labels_test = pd.DataFrame(index=test_data.index, columns=labels)
# for row in test_data.index:
#     birds = test_data.loc[row, 'birds'].split(' ')
#     for bird in birds:
#         df_labels_test.loc[row, bird] = 1
# df_labels_test.fillna(0, inplace=True)
# train_labels = pd.concat([train_labels, df_labels_train], axis=1)
# test_data = pd.concat([test_data, df_labels_test], axis=1)
# df_labels_train.sum().sort_values(ascending=False)[:10]

# print("done")
# # convert to one-hot