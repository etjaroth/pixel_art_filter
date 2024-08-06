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
        
        # Initial image and palette
        self.image_display = None
        self.image_path = tk.StringVar()
        self.reset()
        
        # Creating widgets
        self.background_color = "lightblue"
        self.create_widgets()
        
    # -------------------------------------------------------------------------

    def create_widgets(self):
        # Toggle button for collapsing
        self.toggle_button = tk.Button(self.main_frame, text="Show Controls", command=self.toggle_controls)
        self.toggle_button.grid(row=0, column=0, padx=0, pady=0, sticky="w")
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.main_frame, width=400, height=400)
        self.canvas.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Label for image file path
        self.filepath_label = tk.Label(self.main_frame, textvariable=self.image_path)
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
        
        # Reset
        self.reset_button = tk.Button(self.operation_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=7, column=0, pady=5, sticky="ew")
        
        # Save image button
        self.save_button = tk.Button(self.operation_frame, text="Save Image", command=self.save_image)
        self.save_button.grid(row=8, column=0, pady=5, sticky="ew")
        
        self.update_enables()
    
    # -------------------------------------------------------------------------
    
    def load_image(self):
        # File dialog to select image
        self.image_path.set(filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")]))
        if self.image_path.get():
            self.image = cv2.imread(self.image_path.get())
            self.display_image(self.image)
            
            self.update_enables()
            
    def load_palette(self):
        self.palette_path = filedialog.askopenfilename(filetypes=[("Palette files", "*.gpl")])
        if self.palette_path:
            self.target_palette = palette.FromFilepath(self.palette_path)
            
            self.update_enables()
    
    # -------------------------------------------------------------------------
      
    def apply_pipeline(self):
        self.update_enables(tk.DISABLED)
        self.root.update_idletasks()
        
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
        image: np.ndarray = cv2.imread(self.image_path.get())
        processed_image: np.ndarray = pipeline.apply(image)
        
        # Display
        self.processed_image = processed_image
        self.display_image(processed_image)
        
        self.update_enables()
        
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
        # self.image = image
        image.thumbnail((400, 400))
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_display = self.canvas.create_image(200, 200, image=self.tk_image)
        
    def save_image(self):
        img = self.image
        if self.processed_image is not None:
            img = self.processed_image
        
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
        if save_path:
            cv2.imwrite(save_path, img)
            messagebox.showinfo("Image Saved", f"Image successfully saved to {save_path}")
      
    def toggle_controls(self):
        if self.controls_visible:
            self.operation_frame.grid_remove()
            self.toggle_button.config(text="Show Controls")
        else:
            self.operation_frame.grid()
            self.toggle_button.config(text="Hide Controls")
        
        self.controls_visible = not self.controls_visible
        
        if self.processed_image is not None:
            self.display_image(self.image)
        elif self.image is not None:
            self.display_image(self.processed_image)
      
    def update_enables(self, setenable = None):
        if setenable is None:
            self.update_enables(tk.DISABLED)
            self.load_image_button.config(state=tk.NORMAL)
            self.load_palette_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
        
            if self.image_path.get():
                self.simplify_checkbox.config(state=tk.NORMAL)
                self.add_edges_checkbox.config(state=tk.NORMAL)
                
                self.apply_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL)
                
                if self.palette_path:
                    self.quantize_checkbox.config(state=tk.NORMAL)
                else:
                    self.quantize_var.set(False)
                    
        else:
            self.load_image_button.config(state=setenable)
            self.load_palette_button.config(state=setenable)
            self.reset_button.config(state=setenable)
            
            self.simplify_checkbox.config(state=setenable)
            self.quantize_checkbox.config(state=setenable)
            self.add_edges_checkbox.config(state=setenable)
            
            self.apply_button.config(state=setenable)
            self.save_button.config(state=setenable)
                
    # -------------------------------------------------------------------------
    
    def reset(self):
        self.image: np.ndarray = None
        self.image_path.set("")
        self.processed_image = None
        
        self.target_palette: palette.Palette|None = None
        self.palette_path = ""
        
        if self.image_display is not None:
            self.canvas.delete(self.image_display)
            self.image_display = None
