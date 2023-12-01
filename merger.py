import argparse
import torch
import copy
from safetensors.torch import load_file, save_file

def parse_args():
    parser = argparse.ArgumentParser(description='Merge two models')
    parser.add_argument('--model1', type=str, required=True, help='Path to the first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to the second model')
    parser.add_argument('--output', type=str, required=True, help='Path to save the merged model')
    parser.add_argument('--range', type=float, nargs=2, default=[50, 50], help='Range between model1 and model2')
    return parser.parse_args()

def load_model(model_path):
    state_dict = load_file(model_path) if model_path.endswith('.safetensors') else torch.load(model_path)
    model = torch.nn.Module()
    model.load_state_dict(state_dict)
    model.eval()
    return model

def merge_models(model1, model2, range):
    if type(model1) != type(model2):
        raise ValueError("Models must be of the same class")

    merged = copy.deepcopy(model1)

    with torch.no_grad():
        for (key1, param1), (key2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            if key1 != key2:
                raise ValueError("Models have different architectures")
            merged.state_dict()[key1].copy_((param1 * (1 - range[0]/100) + param2 * range[1]/100))

    return merged

def save_model(model, path):
    if not path.endswith('.safetensors') and not path.endswith('.pth'):
        path = path.rsplit('.', 1)[0] + '.pth'
    if path.endswith('.safetensors'):
        save_file(model, path)
    else:
        torch.save(model.state_dict(), path)

def run():
    args = parse_args()
    model1 = load_model(args.model1)
    model2 = load_model(args.model2)

    merged_model = merge_models(model1, model2, args.range)

    save_model(merged_model, args.output)

if __name__ == '__main__':
    run()

