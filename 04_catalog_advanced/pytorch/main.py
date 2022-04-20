import models.my_network as MyNetwork
import yaml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse


def train_step(images, labels, model, loss_object, optimizer, train_log):
    model.train()

    predictions = model(images)
    loss = loss_object(predictions, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    batch = images.shape[0]

    train_log['loss'] += loss.item() * batch
    train_log['div'] += batch

    _, predicted_labels = torch.max(predictions.data, 1)
    train_log['correct'] += (predicted_labels == labels).sum().item()


def test_step(images, labels, model, loss_object, test_log):
    model.eval()
    with torch.no_grad():
        predictions = model(images)
        loss = loss_object(predictions, labels)

    batch = images.shape[0]

    test_log['loss'] += loss.item() * batch
    test_log['div'] += batch

    _, predicted_labels = torch.max(predictions.data, 1)
    test_log['correct'] += (predicted_labels == labels).sum().item()


def pop_tail(dataset, pop_factor=1):
    sz = dataset.__len__()
    dataset.__dict__['data'] = dataset.__dict__['data'][:sz//pop_factor]
    dataset.__dict__['targets'] = dataset.__dict__['targets'][:sz // pop_factor]


def mk_dataset(cfg):
    batch_size = cfg['hyper_parameters']['batch_size']
    data_scale_factor = cfg['data_scale_factor']

    trainset = torchvision.datasets.MNIST(root='../../00_data', train=True,
                                          download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root='../../00_data', train=False,
                                         download=True, transform=transforms.ToTensor())

    pop_tail(trainset, data_scale_factor)
    pop_tail(testset, data_scale_factor)

    train_ds = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_ds = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    return train_ds, test_ds


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/bn_relu.yaml')
    return parser.parse_args()


def config_print(config, depth=0):
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if type(v) == dict:
            print(*prefix)
            config_print(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = get_parser()
    with open(parser.config) as f:
        config = yaml.safe_load(f)
        config_print(config)

    batch_size = config['hyper_parameters']['batch_size']
    epochs = config['hyper_parameters']['epochs']
    learning_rate = config['hyper_parameters']['learning_rate']

    data_scale_factor = config['data_scale_factor']

    train_ds, test_ds = mk_dataset(config)

    model = MyNetwork.MyModel(config['network_parameters'])
    model.to(device)
    #
    loss_object = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    EPOCHS = epochs
    for epoch in range(EPOCHS):

        train_log = {'loss': 0., 'div': 0, 'correct': 0}
        test_log = {'loss': 0., 'div': 0, 'correct': 0}
        for images, labels in train_ds:
            train_step(images.to(device), labels.to(device), model, loss_object, optimizer, train_log)

        for test_images, test_labels in test_ds:
            test_step(test_images.to(device), test_labels.to(device), model, loss_object, test_log)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_log["loss"] / train_log["div"]}, '
            f'Accuracy: {train_log["correct"] / train_log["div"] * 100}, '
            f'Test Loss: {test_log["loss"] / test_log["div"]}, '
            f'Test Accuracy: {test_log["correct"] / test_log["div"] * 100}'
        )

    print(model)

if __name__ == '__main__':
    main()
