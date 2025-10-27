
import cv2
import numpy as np
import onnxruntime as ort
import argparse

def preprocess(image, input_size=(1024, 1024)):
    """
    Preprocesses the input image for the model.
    """
    # Resize the image to the model's input size
    resized_image = cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)
    # Convert the image to RGB
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Normalize the image
    normalized_image = rgb_image.astype(np.float32) / 255.0
    # Transpose the dimensions to (channels, height, width)
    transposed_image = np.transpose(normalized_image, (2, 0, 1))
    # Add a batch dimension
    input_tensor = np.expand_dims(transposed_image, axis=0)
    return input_tensor, resized_image

def postprocess(output_tensor, original_image):
    """
    Postprocesses the model's output to create a segmentation mask.
    """
    # Get the segmentation mask from the output tensor
    mask = np.squeeze(output_tensor)
    # Convert the mask to a binary image
    mask = (mask > 0.5).astype(np.uint8) * 255
    # Resize the mask to the original image size
    resized_mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Create a 3-channel mask
    mask_3_channel = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    # Overlay the mask on the original image
    overlay = cv2.addWeighted(original_image, 1, mask_3_channel, 0.5, 0)
    return overlay

def main():
    """
    Main function to run the image segmentation.
    """
    parser = argparse.ArgumentParser(description="Image segmentation using ONNX model.")
    parser.add_argument("input_image", help="Path to the input image.")
    parser.add_argument("output_image", help="Path to save the output image.")
    parser.add_argument("--model", default="models/rmbg.onnx", help="Path to the ONNX model.")
    args = parser.parse_args()

    # Load the ONNX model
    session = ort.InferenceSession(args.model)

    # Load the input image
    image = cv2.imread(args.input_image)
    if image is None:
        print(f"Error: Could not read image from {args.input_image}")
        return

    # Preprocess the image
    input_tensor, _ = preprocess(image)

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})

    # Postprocess the output
    result_image = postprocess(output[0], image)

    # Save the output image
    cv2.imwrite(args.output_image, result_image)
    print(f"Segmentation complete. Output saved to {args.output_image}")

if __name__ == "__main__":
    main()
