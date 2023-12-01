import argparse
import torch
import copy
from safetensors.torch import load_file, save_file

class ModelMerger:
    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Merge two models')
        parser.add_argument('--model1', type=str, required=True, help='Path to the first model')
        parser.add_argument('--model2', type=str, required=True, help='Path to the second model')
        parser.add_argument('--output', type=str, required=True, help='Path to save the merged model')
        parser.add_argument('--range', type=float, nargs=2, default=[50, 50], help='Range between model1 and model2')
        return parser.parse_args()

    def load_model(self, model_path):
        # Define your model architecture here
        model_architecture = MyModel()  # Replace 'MyModel' with your actual model class

        if model_path.endswith('.safetensors'):
            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(model_path)
    
        model_architecture.load_state_dict(state_dict)
        model_architecture.eval()
        return model_architecture

    def merge_models(self, model1, model2, range):
        if type(model1) != type(model2):
            raise ValueError("Models must be of the same class")

        merged = copy.deepcopy(model1)

        with torch.no_grad():
            for (key1, param1), (key2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
                if key1 != key2:
                    raise ValueError("Models have different architectures")
                merged.state_dict()[key1].copy_((param1 * (1 - range[0]/100) + param2 * range[1]/100))

        return merged

    def save_model(self, model, path):
        if not path.endswith('.safetensors') and not path.endswith('.pth'):
            path = path.rsplit('.', 1)[0] + '.pth'
        if path.endswith('.safetensors'):
            save_file(model, path)
        else:
            torch.save(model.state_dict(), path)
            

    def run(self):
        model1 = self.load_model(self.args.model1)
        model2 = self.load_model(self.args.model2)

        merged_model = self.merge_models(model1, model2, self.args.range)

        self.save_model(merged_model, self.args.output)

if __name__ == '__main__':
    merger = ModelMerger()
    merger.run()
