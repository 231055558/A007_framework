import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import os
from networks.deeplabv3plus_debug import DeepLabV3PlusClassifierAttentionHeadOutputMerge


class ModelVisualizer(tk.Tk):
    def __init__(self, model=None):
        super().__init__()

        self.title("PyTorch Model Weight Visualizer")
        self.geometry("1000x700")  # 设置初始窗口大小

        # 配置列的权重，使两列平均分配空间
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Model passed as argument
        self.model = model
        self.pth_weights = None

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Left Frame for PyTorch Model
        self.left_frame = tk.Frame(self)
        self.left_frame.grid(row=0, column=0, padx=15, pady=15, sticky='nsew')
        
        # 配置left_frame的网格
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)

        self.left_label = tk.Label(self.left_frame, text="PyTorch Model Layers", font=("Arial", 14))
        self.left_label.grid(row=0, column=0, pady=10)

        # 创建左侧滚动条
        left_scrollbar = tk.Scrollbar(self.left_frame)
        left_scrollbar.grid(row=1, column=1, sticky='ns')

        self.left_listbox = tk.Listbox(self.left_frame, 
                                     font=("Arial", 30),
                                     yscrollcommand=left_scrollbar.set)
        self.left_listbox.grid(row=1, column=0, sticky='nsew')
        left_scrollbar.config(command=self.left_listbox.yview)

        # Right Frame for PTH Weights
        self.right_frame = tk.Frame(self)
        self.right_frame.grid(row=0, column=1, padx=15, pady=15, sticky='nsew')
        
        # 配置right_frame的网格
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        self.right_label = tk.Label(self.right_frame, text="PTH File Weights", font=("Arial", 14))
        self.right_label.grid(row=0, column=0, pady=10)

        # 创建右侧滚动条
        right_scrollbar = tk.Scrollbar(self.right_frame)
        right_scrollbar.grid(row=1, column=1, sticky='ns')

        self.right_listbox = tk.Listbox(self.right_frame, 
                                      font=("Arial", 30),
                                      yscrollcommand=right_scrollbar.set)
        self.right_listbox.grid(row=1, column=0, sticky='nsew')
        right_scrollbar.config(command=self.right_listbox.yview)

        # Buttons Frame
        button_frame = tk.Frame(self)
        button_frame.grid(row=1, column=0, columnspan=2, pady=15)

        self.load_pth_button = tk.Button(button_frame, 
                                       text="Load PTH File", 
                                       command=self.load_pth,
                                       font=("Arial", 12))
        self.load_pth_button.pack(side=tk.LEFT, padx=5)

        self.compare_button = tk.Button(button_frame,
                                      text="Compare Weights",
                                      command=self.compare_weights,
                                      font=("Arial", 12))
        self.compare_button.pack(side=tk.LEFT, padx=5)

        # Display model information if model is already passed
        if self.model is not None:
            self.display_model_info()

    def load_pth(self):
        # Ask for the PTH file
        pth_file = filedialog.askopenfilename(filetypes=[("PTH Files", "*.pth")])
        if not pth_file:
            return

        # Load PTH weights
        try:
            self.pth_weights = torch.load(pth_file)
            self.display_pth_info()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading PTH file: {str(e)}")

    def display_model_info(self):
        if self.model is None:
            return

        self.left_listbox.delete(0, tk.END)
        for name, param in self.model.named_parameters():
            self.left_listbox.insert(tk.END, f"{name}: {param.shape}")

    def display_pth_info(self):
        if self.pth_weights is None:
            return

        self.right_listbox.delete(0, tk.END)
        for name, param in self.pth_weights.items():
            self.right_listbox.insert(tk.END, f"{name}: {param.shape}")

    def compare_weights(self):
        if self.model is None or self.pth_weights is None:
            return

        mismatches = []
        for name, param in self.model.named_parameters():
            if name in self.pth_weights:
                if param.shape != self.pth_weights[name].shape:
                    mismatches.append(
                        f"Shape mismatch: {name} - Model: {param.shape}, PTH: {self.pth_weights[name].shape}")
            else:
                mismatches.append(f"Missing in PTH: {name}")

        if mismatches:
            messagebox.showwarning("Mismatch Found", "\n".join(mismatches))
        else:
            messagebox.showinfo("Match Found", "All model parameters match with the PTH file.")


if __name__ == "__main__":
    from networks.cross_field_transformer import CrossFieldTransformer

    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)

    # 定义模型参数
    depth = 50  # 支持 50, 101, 152
    in_channels = 3
    num_classes = 10
    batch_size = 2
    image_size = 224  # 假设输入图像大小为 224x224

    # 创建模型实例
    model = CrossFieldTransformer(num_classes=8)
    app = ModelVisualizer(model=model)
    app.mainloop()
