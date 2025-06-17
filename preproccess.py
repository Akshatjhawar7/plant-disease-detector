import os
import matplotlib.pyplot as plt
from collections import Counter
import random
import cv2

# Define the directory paths
train_dir = r"TRAIN_DATASET"
valid_dir = r"VALIDATION_DATASET"

def get_class_distribution(data_dir):
    class_counts = Counter()
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            class_counts[class_name] = num_images
    return class_counts

# Get distributions
train_distribution = get_class_distribution(train_dir)
valid_distribution = get_class_distribution(valid_dir)

# Plot distributions
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].bar(train_distribution.keys(), train_distribution.values())
ax[0].set_title('Training Set Class Distribution')
ax[0].set_xticklabels(train_distribution.keys(), rotation=90)

ax[1].bar(valid_distribution.keys(), valid_distribution.values())
ax[1].set_title('Validation Set Class Distribution')
ax[1].set_xticklabels(valid_distribution.keys(), rotation=90)

plt.tight_layout()
plt.show()

# Sample a few images from each class to check size
def check_image_sizes(data_dir, sample_size=5):
    sizes = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            image_files = os.listdir(class_path)
            samples = random.sample(image_files, min(sample_size, len(image_files)))
            for img_name in samples:
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                sizes.append(img.shape[:2])  # Append height and width
    return sizes

train_image_sizes = check_image_sizes(train_dir)
print("Sample of image sizes in training set:", train_image_sizes)
def plot_random_images(data_dir, classes_to_display=5, images_per_class=3):
    fig, axes = plt.subplots(classes_to_display, images_per_class, figsize=(12, 10))
    fig.suptitle("Random Images Across Classes", fontsize=16)
    class_names = os.listdir(data_dir)
    
    for i, class_name in enumerate(random.sample(class_names, classes_to_display)):
        class_path = os.path.join(data_dir, class_name)
        image_files = os.listdir(class_path)
        
        for j in range(images_per_class):
            img_path = os.path.join(class_path, random.choice(image_files))
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(class_name, size=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Display images
plot_random_images(train_dir)

def plot_pixel_intensity_histogram(data_dir, sample_size=100):
    intensities = []
    for class_name in random.sample(os.listdir(data_dir), 5):  # Limit to a few classes
        class_path = os.path.join(data_dir, class_name)
        image_files = random.sample(os.listdir(class_path), min(sample_size, len(os.listdir(class_path))))
        
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            intensities.extend(img.ravel())  # Flatten to one-dimensional
    
    plt.hist(intensities, bins=50, color='blue', alpha=0.7)
    plt.title('Pixel Intensity Distribution Across Sampled Images')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()

# Plot pixel intensity histogram
plot_pixel_intensity_histogram(train_dir)
