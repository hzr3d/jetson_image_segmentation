
# Jetson Image Segmentation

This project provides a simple way to perform image segmentation on a Jetson device using an ONNX model.

## Setup

1.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Add an image:**

    Place an image file named `sample_image.jpg` in the `jetson_image_segmentation` directory.

## Usage

To run the image segmentation, use the following command:

```bash
python segment_image.py sample_image.jpg output.jpg
```

This will process `sample_image.jpg` and save the segmented image as `output.jpg`.

### Command-line arguments

*   `input_image`: Path to the input image.
*   `output_image`: Path to save the output image.
*   `--model`: (Optional) Path to the ONNX model. Defaults to `models/rmbg.onnx`.
