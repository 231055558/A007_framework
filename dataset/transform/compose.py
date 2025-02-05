class Compose:
    def __init__(self, transforms=None):
        self.transforms = transforms if transforms is not None else []

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def add_transform(self, transform):
        self.transforms.append(transform)

    def remove_transform(self, transform):
        self.transforms.remove(transform)

    def clear(self):
        self.transforms = []