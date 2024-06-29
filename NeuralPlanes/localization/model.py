from torch import nn
from transformers import AutoImageProcessor, AutoBackbone
import torch
import numpy as np

class Dino(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        out_features = ["stage2", "stage5", "stage8", "stage11"]
        model = AutoBackbone.from_pretrained("facebook/dinov2-base", out_features=out_features)

        self.processor = processor
        self.model = model

        in_features = model.config.hidden_size * len(out_features)
        patch = 16

        self.patch = patch
        self.num_features = num_features

        self.upsample = nn.Sequential(
            nn.Conv2d(in_features, (num_features*patch**2), (1,1)),
            nn.LeakyReLU(),
            nn.Conv2d((num_features*patch**2), (num_features*patch**2), (1, 1)),
            nn.LeakyReLU()
        )

    def preprocess_batch(self, data):
        batch, height, width, c = data["image"].shape
        return { **data, "orig_shape": (height,width), "inputs": self.processor(images=data["image"], return_tensors="pt") }

    def forward(self, data):
        patch, num_features = self.patch, self.num_features

        print(data["inputs"])

        outputs = self.model(**data["inputs"])
        orig_shape = data["orig_shape"]

        feature_maps = outputs.feature_maps  # torch.Size([batch, 3072, 16, 16])

        patch_features = torch.cat(feature_maps, dim=1)
        batch, c, height, width = patch_features.shape

        features = self.upsample(patch_features)
        features = features.reshape(batch, num_features, patch, patch, height, width)
        features = features.permute(0, 1, 4, 2, 5, 3)
        features = features.reshape(batch, num_features, height*patch, width*patch)

        return { **data, "image": features }

if __name__ == "__main__":
    from NeuralPlanes.pipeline.dataloader import collate_fn

    model = Dino(16)
    img = np.zeros((224,224,3))
    data = collate_fn(preprocess_batch=model.preprocess_batch)([{"image": img}])
    
    features = model(data)["image"]
    print(features.shape)