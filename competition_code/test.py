import argparse
import torch
from torch import nn
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import data_loader.datasets as module_dataset
import data_loader.preprocessors as module_preprocessor
import librosa.display as lbd
from torchvision.utils import save_image
import os
import pandas as pd
from resnest.torch import resnest50

def load_net(checkpoint_path, num_classes=397, device = torch.device("cpu")):
    net = resnest50(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    dummy_device = torch.device("cpu")
    d = torch.load(checkpoint_path, map_location=dummy_device)
    for key in list(d.keys()):
        d[key.replace("model.", "")] = d.pop(key)
    net.load_state_dict(d)
    net = net.to(device)
    net = net.eval()
    return net

def main(config):
    logger = config.get_logger('test')
    preprocessor = config.init_obj('preprocessor', module_preprocessor)
    #test_dataset = config.init_obj('dataset', module_dataset, preprocessor, mode = 'xeno', vanilla = True)
    test_dataset = config.init_obj('dataset', module_dataset, preprocessor, mode = 'soundscape', vanilla = True)
    data_loader = config.init_obj('data_loader', module_data, test_dataset)
    # train_dataset = config.init_obj('dataset', module_dataset, preprocessor, mode = 'xeno', vanilla = True)
    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

    external_model = True
    # build model architecture
    if not external_model:
        model = config.init_obj('arch', module_arch)
        logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if external_model:
        model = load_net(config.resume, 397, device)
    else:
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        model = model.to(device)
        model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)
    dir = "image_samples"

    column_names = ["Predictions", "Labels"]
    outputs = []
    targets = []
    metrics = [0, 0, 0, 0]
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            #lbd.specshow(data)
            data = torch.squeeze(data, dim=0)
            # data = torch.flip(data, [2])
            target = torch.squeeze(target, dim=0)
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.sigmoid(output)
            #outputs.append(output)
            #targets.append(target)

            # save sample images, or do something with output here
            #
            targets.append(pd.DataFrame(test_dataset.prediction_string(output, target)))
            call_idx = torch.argmax(target, dim=1)
            for j in range(target.shape[0]):
                if call_idx[j] > 0:
                    fn = 'img' + str(i) + '-' + str(j) + '.png'
                    save_image(data[j], os.path.join(dir, fn))
            #
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
                #metrics += metric(output, target) * batch_size
            #metrics += metric_fns[0](output, target)

    df = pd.concat(targets)
    df.to_csv('test_results.csv')
    print(metrics)
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
