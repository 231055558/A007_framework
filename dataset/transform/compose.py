# from time import time
from torchvision.transforms import Compose
# class Compose:
#     def __init__(self, transforms=None):
#         self.transforms = transforms if transforms is not None else []
#     def __call__(self, data):
#         for t in self.transforms:
#             self.start_time = time()
#             data = t(data)
#             print(f"cost_time: {time() - self.start_time},name:{t}")
#         print("--------------------------------------------------")
#         return data
#
#     def add_transform(self, transform):
#         self.transforms.append(transform)
#
#     def remove_transform(self, transform):
#         self.transforms.remove(transform)
#
#     def clear(self):
#         self.transforms = []