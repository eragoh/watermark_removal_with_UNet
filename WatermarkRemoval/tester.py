# import torch
# from unet_model import UNet  # Import the UNet model class
# from PIL import Image
# from torchvision import transforms

# # Initialize the model and load the trained weights
# model = UNet(n_channels=3, n_classes=3)  # Adjust parameters as per your model
# model.load_state_dict(torch.load('model_epoch_1.pth'))
# model.eval()  # Set the model to evaluation mode

# # Define the transformation
# transform = transforms.Compose([transforms.ToTensor()])

# # Load and transform the image
# input_image_path = '../Data/CLWD/train/Watermarked_image/18.png'
# input_image = Image.open(input_image_path)
# input_image = transform(input_image)
# input_image = input_image.unsqueeze(0)  # Add batch dimension

# # Assuming you're using a CPU for inference
# output = model(input_image)

# # Convert the output tensor to an image
# output_image = transforms.ToPILImage()(output.squeeze(0))

# # Save the output image
# output_image_path = '../Data/CLWD/train/Watermarked_image/new18.png'
# output_image.save(output_image_path)


import torch
from unet_model import UNet  # Import the UNet model class
from PIL import Image
from torchvision import transforms

# Initialize the model and load the trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=3).to(device)  # Move model to the correct device
model.load_state_dict(torch.load('model_epoch_1.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([transforms.ToTensor()])
# Include other necessary transforms here (e.g., normalization)

# Load and transform the image
input_image_path = '../Data/CLWD/train/Watermarked_image/18.jpg'
input_image = Image.open(input_image_path)
input_image = transform(input_image)
input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Perform inference
output = model(input_image)

# Move output back to CPU and convert to PIL Image
output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

# Save the output image
output_image_path = 'results/new18.jpg'
output_image.save(output_image_path)
