#new new
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variables to store the current state
filtered_image_global = None
canvas = None
fig = None
axs = None

# Function to load a satellite image and handle multi-band images
def load_image(filepath):
    if filepath.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        import cv2
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image.ndim == 2:  # Grayscale
            image = np.expand_dims(image, axis=0)
        else:  # Convert RGB to bands
            image = np.transpose(image, (2, 0, 1))
    elif filepath.lower().endswith(".tif"):
        with rasterio.open(filepath) as src:
            image = src.read()  # Reads all bands
    else:
        raise ValueError("Unsupported file format!")
    return image

# Fourier Transform function for multi-band images
def fourier_transform_image(image):
    transformed = []
    for band in image:
        f_transform = np.fft.fft2(band)
        f_shift = np.fft.fftshift(f_transform)
        transformed.append(f_shift)
    return transformed

# Apply inverse Fourier Transform to reconstruct the image
def inverse_fourier_transform_image(filtered_bands):
    reconstructed = []
    for band in filtered_bands:
        f_ishift = np.fft.ifftshift(band)
        img_back = np.fft.ifft2(f_ishift)
        reconstructed.append(np.abs(img_back))
    return np.stack(reconstructed, axis=0)

# Gaussian and Butterworth filter functions
def gaussian_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            mask[u, v] = np.exp(-(d ** 2) / (2 * (cutoff ** 2)))
    return mask

def butterworth_low_pass_filter(shape, cutoff, order):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            d = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            mask[u, v] = 1 / (1 + (d / cutoff) ** (2 * order))
    return mask

# Apply low-pass filters on each band
def apply_low_pass_filter(f_shifted_bands, filter_type, cutoff, order=2):
    filtered_bands = []
    for f_shift in f_shifted_bands:
        if filter_type == "gaussian":
            filter_mask = gaussian_low_pass_filter(f_shift.shape, cutoff)
        elif filter_type == "butterworth":
            filter_mask = butterworth_low_pass_filter(f_shift.shape, cutoff, order)
        filtered_band = f_shift * filter_mask
        filtered_bands.append(filtered_band)
    return filtered_bands

# Tkinter UI functions
def browse_file():
    global filepath
    filepath = filedialog.askopenfilename(
        filetypes=[
            ("Image files", "*.tif *.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*"),
        ]
    )
    if filepath:
        file_label.config(text=f"Selected: {filepath.split('/')[-1]}")
    else:
        file_label.config(text="No file selected.")

def process_image():
    global filtered_image_global, canvas, fig, axs
    try:
        if not filepath:
            messagebox.showerror("Error", "No file selected!")
            return

        # Load and process image
        image = load_image(filepath)
        f_shifted_bands = fourier_transform_image(image)

        # Get filter type, cutoff, and order
        filter_type = filter_var.get()
        cutoff = int(cutoff_entry.get())
        order = int(order_entry.get()) if filter_type == "butterworth" else 0

        # Apply selected filter
        filtered_bands = apply_low_pass_filter(
            f_shifted_bands, filter_type, cutoff, order
        )
        filtered_image_global = inverse_fourier_transform_image(filtered_bands)

        # Display results dynamically
        if canvas:
            canvas.get_tk_widget().destroy()
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        display_results(image, f_shifted_bands, filtered_bands, filtered_image_global)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_results(image, f_shifted_bands, filtered_bands, filtered_image):
    global canvas, fig, axs
    # Original Image
    axs[0, 0].imshow(image[0], cmap="gray")
    axs[0, 0].set_title("Original Image (First Band)")
    axs[0, 0].axis("off")

    # Fourier Transform Magnitude
    original_magnitude = np.log(np.abs(f_shifted_bands[0]) + 1)
    axs[0, 1].imshow(original_magnitude, cmap="gray")
    axs[0, 1].set_title("Fourier Transform (Magnitude)")
    axs[0, 1].axis("off")

    # Filtered Image
    axs[1, 0].imshow(filtered_image[0], cmap="gray")
    axs[1, 0].set_title("Filtered Image (First Band)")
    axs[1, 0].axis("off")

    # Filtered Fourier Transform Magnitude
    filtered_magnitude = np.log(np.abs(filtered_bands[0]) + 1)
    axs[1, 1].imshow(filtered_magnitude, cmap="gray")
    axs[1, 1].set_title("Filtered Fourier Transform (Magnitude)")
    axs[1, 1].axis("off")

    # Display the plot in a Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

def save_image():
    global filtered_image_global
    if filtered_image_global is None:
        messagebox.showerror("Error", "No image to save!")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
    )
    if filepath:
        plt.imsave(filepath, filtered_image_global[0], cmap="gray")
        messagebox.showinfo("Success", "Image saved successfully!")

# Tkinter GUI setup
window = tk.Tk()
window.title("Image Filtering with Fourier Transform")
window.geometry("600x400")

# File selection
file_label = tk.Label(window, text="No file selected.")
file_label.pack(pady=10)

browse_button = tk.Button(window, text="Browse File", command=browse_file)
browse_button.pack(pady=5)

# Filter selection
filter_var = tk.StringVar(value="gaussian")
tk.Label(window, text="Select Filter Type:").pack(pady=5)
tk.Radiobutton(window, text="Gaussian", variable=filter_var, value="gaussian").pack()
tk.Radiobutton(window, text="Butterworth", variable=filter_var, value="butterworth").pack()

# Cutoff frequency input
tk.Label(window, text="Cutoff Frequency:").pack(pady=5)
cutoff_entry = tk.Entry(window)
cutoff_entry.pack()

# Order input (only for Butterworth)
tk.Label(window, text="Order (Butterworth only):").pack(pady=5)
order_entry = tk.Entry(window)
order_entry.pack()

# Process button
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.pack(pady=5)

# Save button
save_button = tk.Button(window, text="Save Image", command=save_image)
save_button.pack(pady=20)

# Run the application
window.mainloop()
