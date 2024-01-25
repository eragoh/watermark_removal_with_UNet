import torch
from unet_model import UNet  # Import the UNet model class
from PIL import Image
from torchvision import transforms

# Initialize the model and load the trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=3).to(device)  # Move model to the correct device
model.load_state_dict(torch.load('ready_models/model1-20epochs.pth', map_location=device))
model.eval()  # Set the model to evaluation mode

# Define the transformation
transform = transforms.Compose([transforms.ToTensor()])

# Load and transform the image
for i in range(50,60):
    input_image_path = '../Data/CLWD/test/Watermarked_image/' + str(i) + '.jpg'
    input_image = Image.open(input_image_path)
    input_image = transform(input_image)
    input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Perform inference
    output = model(input_image)

    # Move output back to CPU and convert to PIL Image
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

    # Save the output image
    output_image_path = 'results/results/' + str(i) + '-tested.jpg'
    output_image.save(output_image_path)
