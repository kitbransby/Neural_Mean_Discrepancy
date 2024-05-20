import torch
import torch.nn as nn
from tqdm import tqdm

class Neural_Mean_Discrepancy(nn.Module):
    def __init__(self, model, layer_names, device):
        super().__init__()
        self.model = model
        self.model.eval()
        self.device = device
        self.layer_names = layer_names
        self.activation = {}
        # register the hooks using the requested layer names
        self.register_activations()

    def get_activations(self, name):
        # https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/6
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def register_activations(self):
        # register a forward hook for every requested layer name
        for name, layer in self.model.named_modules():
            if name in self.layer_names:
                layer.register_forward_hook(self.get_activations(name))

    def fit_in_distribution_dataset(self, id_dataset):
        """
        :param id_dataset: a torch.Dataset() where getitem outputs a single image torch.Tensor([C, H, W]) and label ()
        """
        print('Fitting in-distribution dataset..')
        _, self.nmf = self.compute_activations(id_dataset)

    def predict_nmd_unk_distribtion(self, ud_dataset):
        """
        :param ud_dataset: a list of images as [torch.Tensor(),...], or a torch.Dataset()
        :return:
        """
        print('Predicting nmd of unknown distribution dataset..')
        nmf_per_sample, _ = self.compute_activations(ud_dataset)

        nmd_score, nmd_per_sample = self.nmd_score(nmf_per_sample)

        return nmd_score, nmd_per_sample

    def compute_activations(self, dataset):

        # create empty dictionary to store activations for every example
        layer_activations = {key: [] for key in self.layer_names}

        # iterate through dataset
        for (x, y) in tqdm(dataset):

            # pass single image (1, C, H, W) through model
            _ = self.model(x.to(self.device).unsqueeze(0))

            # iterate through all the layers we need activations for
            for layer_name in self.layer_names:

                # get activation map
                activation_map = self.activation[layer_name]

                # take mean over the spatial dims
                channel_activations = activation_map.mean(dim=[0, 2, 3])

                # append to layer activation dictionary
                layer_activations[layer_name].append(channel_activations)

        # stack activations per sample
        activations_per_sample = torch.cat(
            [torch.stack(activations, dim=0) for layer_name, activations in layer_activations.items()],
            dim=1)

        # take mean over all examples, and concat to a single vector (neural mean feature)
        nmf = torch.cat(
            [torch.stack(activations, dim=0).mean(dim=0) for layer_name, activations in layer_activations.items()],
            dim=0)

        return activations_per_sample, nmf

    def nmd_score(self, activations):

        #print(activations.shape, self.nmf.shape)
        nmd = activations - self.nmf
        nmd_reduced = nmd.mean(dim=0)

        return nmd_reduced, nmd



