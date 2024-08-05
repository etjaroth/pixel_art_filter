import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

import palette
import filters
from converter_pipeline import ConverterPipeline
import helper_functions

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Creating main container frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=3)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Creating widgets
        self.background_color = "lightblue"
        self.create_widgets()
        
        # Initial image and image path
        self.image = None
        self.image_path = ""
        self.processed_image = None

    def create_widgets(self):
        # Toggle button for collapsing
        self.toggle_button = tk.Button(self.main_frame, text="Show Controls", command=self.toggle_controls)
        self.toggle_button.grid(row=0, column=0, padx=0, pady=0, sticky="w")
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.canvas.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Label for image file path
        self.filepath_label = tk.Label(self.main_frame, text="")
        self.filepath_label.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")
        
        # Control visibility flag
        self.controls_visible = False

        # Initialize operation frame
        self.init_operation_frame()
        
        self.toggle_controls()  # Start with controls hidden

    def init_operation_frame(self):
        # Frame for operations with different background color
        self.operation_frame = tk.Frame(self.main_frame, bg=self.background_color)
        self.operation_frame.grid(row=1, column=0, padx=10, pady=0, sticky="nsew")
        
        # Configure the operation frame grid
        self.operation_frame.grid_columnconfigure(0, weight=1)
        
        # Load image button
        self.load_image_button = tk.Button(self.operation_frame, text="Load Image", command=self.load_image)
        self.load_image_button.grid(row=0, column=0, pady=5, sticky="ew")
        
        # Load palette button
        self.load_palette_button = tk.Button(self.operation_frame, text="Load Palette", command=self.load_palette)
        self.load_palette_button.grid(row=1, column=0, pady=5, sticky="ew")
        
        # Processing step enable/disable
        self.operations_label = tk.Label(self.operation_frame, text="Operations", bg="lightblue")
        self.operations_label.grid(row=2, column=0, sticky="w")
        
        # Checkbuttons
        self.simplify_var = tk.BooleanVar()
        self.quantize_var = tk.BooleanVar()
        self.add_edges_var = tk.BooleanVar()
        
        self.simplify_checkbox = tk.Checkbutton(self.operation_frame, text="Simplify", variable=self.simplify_var, bg=self.background_color)
        self.quantize_checkbox = tk.Checkbutton(self.operation_frame, text="Quantize", variable=self.quantize_var, bg=self.background_color)
        self.add_edges_checkbox = tk.Checkbutton(self.operation_frame, text="Add Edges", variable=self.add_edges_var, bg=self.background_color)
        
        self.simplify_checkbox.grid(row=3, column=0, sticky="w")
        self.quantize_checkbox.grid(row=4, column=0, sticky="w")
        self.add_edges_checkbox.grid(row=5, column=0, sticky="w")
        
        # Settings
        self.target_palette = None
        self.size_x_var     = tk.IntVar(value=128)
        self.size_y_var     = tk.IntVar(value=128)
        self.p_var          = tk.IntVar(value=1)
        
        # Apply button
        self.apply_button = tk.Button(self.operation_frame, text="Apply", command=self.apply_pipeline)
        self.apply_button.grid(row=6, column=0, pady=5, sticky="ew")
        
        # Save image button
        self.save_button = tk.Button(self.operation_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=7, column=0, pady=5, sticky="ew")
    
    def toggle_controls(self):
        if self.controls_visible:
            self.operation_frame.grid_remove()
            self.toggle_button.config(text="Show Controls")
        else:
            self.operation_frame.grid()
            self.toggle_button.config(text="Hide Controls")
        self.controls_visible = not self.controls_visible
    
    def load_image(self):
        # File dialog to select image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.image: np.ndarray = cv2.imread(self.image_path)
            self.display_image(self.image)
            self.filepath_label.config(text=self.image_path)
            
    def load_palette(self):
        self.palette_path = filedialog.askopenfilename(filetypes=[("Palette files", "*.gpl")])
        if self.palette_path:
            self.target_palette = palette.FromFilepath(self.palette_path)
        
    def display_image(self, image: np.ndarray):
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        image_width, image_height, _ = image.shape
        image_aspect_ratio = image_width / image_height
        canvas_aspect_ratio = canvas_width / canvas_height
        
        if canvas_aspect_ratio > image_aspect_ratio:
            new_height = canvas_height
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_width = canvas_width
            new_height = int(new_width / image_aspect_ratio)
            
        image = helper_functions.resize_no_interpolation(image, new_width, new_height)
        
        # Convert image to displayable format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        self.image = image
        self.image.thumbnail((400, 400))
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.create_image(200, 200, image=self.tk_image)
    
    def apply_pipeline(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        # Init pipeline
        pipeline: ConverterPipeline = ConverterPipeline(
            self.target_palette,
            self.size_x_var.get(),
            self.size_y_var.get(),
            self.p_var.get(),
            self.add_edges_var.get(),
        )
        
        # Simplify
        if self.simplify_var.get():
            pipeline.append(filters.simplify_img)
        
        # Quantize
        if self.quantize_var.get():
            pipeline.append(filters.quantize_img)
            
        # Remap
        pipeline.append(filters.remap_img)
        
        # Apply
        image: np.ndarray = cv2.imread(self.image_path)
        processed_image: np.ndarray = pipeline.apply(image)
        
        # Display
        self.display_image(processed_image)
    
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("No Processed Image", "Please apply an operation to the image first.")
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, self.processed_image)
            messagebox.showinfo("Image Saved", f"Image successfully saved to {save_path}")
