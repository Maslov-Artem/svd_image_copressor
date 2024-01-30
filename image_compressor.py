import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage import io


def main() -> None:
    st.write(
        """
        # Image Compression

        This Streamlit app allows you to upload an image, compress it and visualize the original and compressed images side by side.
        App is using SVD method for compression

        ## Instructions:
        1. Upload an image by using the "Choose an image" button.
        2. Adjust the 'k' value using the slider on the sidebar. Note that less k value results in a more compressed image.


        ## Details:
        - Singular Value Decomposition (SVD) is applied to the uploaded image.
        - The 'k' value determines the number of singular values used in the reconstruction, affecting the level of compression.
       """
    )
    uploaded_image = st.file_uploader(label="Choose an image", type="jpg")

    if uploaded_image:
        image = io.imread(uploaded_image)
        converted_image = image_conversion(image)

        # Display original and converted images side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_title("Original Image")
        ax1.imshow(image)
        ax1.axis("off")

        ax2.set_title("Converted Image")
        ax2.imshow(converted_image, cmap="gray")  # assuming grayscale
        ax2.axis("off")

        st.pyplot(fig)


def image_conversion(image):
    k = get_k(image.shape[1])

    converted_image = np.zeros_like(image)

    for image_channel in range(image.shape[2]):
        U, sign_val, V = np.linalg.svd(image[:, :, image_channel])

        sigma = np.zeros_like(image[:, :, image_channel], dtype="float64")
        np.fill_diagonal(sigma, sign_val)

        trunc_U = U[:, :k]
        trunc_sigma = sigma[:k, :k]
        trunc_V = V[:k, :]

        converted_channel = trunc_U @ trunc_sigma @ trunc_V
        converted_image[:, :, image_channel] = converted_channel

    return converted_image


def get_k(max_val):
    val = st.sidebar.slider(
        min_value=0,
        max_value=max_val,
        value=max_val,
        label="Choose k value",
        step=-1,
    )
    return val


if __name__ == "__main__":
    main()
