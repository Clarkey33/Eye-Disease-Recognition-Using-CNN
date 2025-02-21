import streamlit as st
import torch
from PIL import Image
from torchvision import transforms, models

# Define the classes
classes = [
    "Cataract",
    "Diabetes",
    "Glaucoma",
]


# @st.cache_resource
def load_resnet_model():
    # Initialize the ResNet-50 model architecture
    resnet_model = models.resnet50(pretrained=False)
    num_classes = len(classes)
    resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, num_classes)

    # Load the saved state dictionary
    resnet_model.load_state_dict(torch.load('models/resnet50_best.pth', map_location=torch.device('cpu')))

    print('Resnet50 loaded successfully')
    return resnet_model


# Load the model
resnet_model = load_resnet_model()


def classify_eye(model, image):
    model = model.eval()

    # Define image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Preprocess the image
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    image = image_transforms(image).unsqueeze(0)  # Add batch dimension

    # Move model and image to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    return classes[predicted.item()]


st.title("Eye Image Classification")

uploaded_file = st.file_uploader("Upload an image of an eye", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button('Classify Image'):
        result = classify_eye(resnet_model, image)
        st.write(f"Prediction: {result}")
