import os 
from torchvision.datasets import VisionDataset
from PIL import Image 
from torchvision import transforms

class CustomDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform= None):
        super(CustomDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root 
        self.classes, self.class_idx = self._find_classes()
        self.samples = self._make_dataset()

    def _find_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return classes, class_idx

    def _make_dataset(self):
        samples = []
        for target_class in sorted(self.class_idx.keys()):
            class_index = self.class_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    samples.append(item)
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

