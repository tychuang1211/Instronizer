import torch
from pathlib import Path
import argparse

# Absolute imports
from classifier.models.mobilenet import MobileNet
from classifier import dataset_loader as dl
from classifier.utils.printing_functions import print_execution_time

@print_execution_time
def load_data_from_folder(path):
    dataset = dl.SpecFolder(path, direct=True)
    # There are 3 spectrograms from 6s window(assuming 1.5s overlap by default)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=1,
                                       shuffle=False,
                                       num_workers=1)


def run(input, checkpoint_path):
    checkpoint_path = '/app/checkpoints/mobilenet__YT_dataset__3s_excerpts.pth.tar'
    model = MobileNet(num_classes=6)
    # Map storage to cpu
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print('[lightweight_classifier.py] using checkpoint: \n{}'.format(checkpoint_path))

    # Load model state
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cpu()
    
    validation_data = load_data_from_folder(input)
    aggregated_output = None

    # Switch model to evaluate mode (validatin/testing)
    # Extremely important!
    model.eval()
    
    # This loop executes once
    for step, (input, target) in enumerate(validation_data):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute output
        output = model(input_var)

        # Sum outputs for each instrument
        aggregated_output = torch.sum(output, dim=0)  # size = [1, ncol]

        max_value, max_value_idx = aggregated_output.max(0)

        print('Output: ', output.data)
        print('Instrument class-wise activation sum averaged', aggregated_output)
        print('Max: {}, instrument_idx: {}'.format(max_value, max_value_idx))
    # Arithmetic average, to preserve softmax output
    return (aggregated_output.data).cpu().numpy().reshape(-1).tolist()


def load(checkpoint_path):
    model = MobileNet(num_classes=6)
    # Map storage to cpu
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    print('[lightweight_classifier.py] using checkpoint: \n{}'.format(checkpoint_path))

    # Load model state
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    model.cpu()

    # Switch model to evaluate mode (validatin/testing)
    # Extremely important!
    model.eval()   

    return model


def get_prediction(model, input):
    validation_data = load_data_from_folder(input)
    aggregated_output = None
    # This loop executes once
    for step, (input, target) in enumerate(validation_data):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute output
        output = model(input_var)

        # Sum outputs for each instrument
        aggregated_output = torch.sum(output, dim=0)  # size = [1, ncol]

        max_value, max_value_idx = aggregated_output.max(0)

        print('Output: ', output.data)
        print('Instrument class-wise activation sum averaged', aggregated_output)
        print('Max: {}, instrument_idx: {}'.format(max_value, max_value_idx))
    # Arithmetic average, to preserve softmax output
    return (aggregated_output.data).cpu().numpy().reshape(-1).tolist()