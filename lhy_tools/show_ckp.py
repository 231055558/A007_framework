import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import os


class ModelVisualizer(tk.Tk):
    def __init__(self, model=None):
        super().__init__()

        self.title("PyTorch Model Weight Visualizer")
        self.geometry("1000x700")  # 增大整个窗口的尺寸

        # Model passed as argument
        self.model = model
        self.pth_weights = None

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Left Frame for PyTorch Model
        self.left_frame = tk.Frame(self, width=500, height=700)  # 增大左侧的宽度
        self.left_frame.grid(row=0, column=0, padx=15, pady=15)
        self.left_frame.grid_propagate(False)

        self.left_label = tk.Label(self.left_frame, text="PyTorch Model Layers", font=("Arial", 14))  # 增大字体
        self.left_label.pack(pady=10)

        self.left_listbox = tk.Listbox(self.left_frame, width=50, height=30, font=("Arial", 12))  # 增大字体并调整列表框大小
        self.left_listbox.pack()

        # Right Frame for PTH Weights
        self.right_frame = tk.Frame(self, width=500, height=700)  # 增大右侧的宽度
        self.right_frame.grid(row=0, column=1, padx=15, pady=15)
        self.right_frame.grid_propagate(False)

        self.right_label = tk.Label(self.right_frame, text="PTH File Weights", font=("Arial", 14))  # 增大字体
        self.right_label.pack(pady=10)

        self.right_listbox = tk.Listbox(self.right_frame, width=50, height=30, font=("Arial", 12))  # 增大字体并调整列表框大小
        self.right_listbox.pack()

        # Buttons
        self.load_pth_button = tk.Button(self, text="Load PTH File", command=self.load_pth,
                                         font=("Arial", 12))  # 增大按钮字体
        self.load_pth_button.grid(row=1, column=1, padx=10, pady=15)

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
    from networks.visiontransformer import VisionTransformer

    model = VisionTransformer(arch='base',
                              img_size=224,
                              patch_size=32,
                              num_classes=8,
                              drop_rate=0.1)
    app = ModelVisualizer(model=model)
    app.mainloop()
