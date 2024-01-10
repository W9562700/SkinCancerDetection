import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model
model_path = "Skin Cancer model (Residual ShieldNet).h5"
model = tf.keras.models.load_model(model_path)
classes = {4: ('nv', ' melanocytic nevi'),
           6: ('mel', 'melanoma'),
           2 :('bkl', 'benign keratosis-like lesions'),
           1:('bcc' , ' basal cell carcinoma'),
           5: ('vasc', ' pyogenic granulomas and hemorrhage'),
           0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
           3: ('df', 'dermatofibroma')}


def preprocess_image(image):
    # Resize the image to match the model's expected input size
    image = cv2.resize(image, (28, 28))
    image = image.reshape((-1,28,28,3))

    return image


# Function to make predictions
def make_prediction(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    rounded_prediction = np.round(prediction, 3)
    
    return classes[np.argmax(rounded_prediction)]

# Streamlit UI
def main():
    st.title("Skin Cancer Prediction App")

    # Navigation bar
    page = st.sidebar.selectbox("Select a page", ["Home", "Camera", "Upload"])

    if page == "Home":
        st.subheader("How to Run this Web App:")
        st.write(
            "1. Navigate to the directory containing this script in your terminal."
        )
        st.write(
            "2. Run the following command to start the Streamlit app: `streamlit run app.py`."
        )
        st.write("3. Open the provided URL in your browser.")

        st.subheader("Features:")
        st.write("- Predict skin cancer using the proposed Residual ShieldNet Model.")
        st.write("- Choose between uploading an image or using live camera feed.")

    elif page == "Camera":
        st.subheader("Live Camera Feed")
        camera = cv2.VideoCapture(0)

        if st.button("Capture Image"):
            _, frame = camera.read()
            st.image(frame, channels="BGR", caption="Captured Image", use_column_width=True)
            prediction_button = st.button("Get Prediction")

            if prediction_button:
                prediction = make_prediction(frame)
                st.write("Prediction:", prediction)

    elif page == "Upload":
        st.subheader("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
            prediction_button = st.button("Get Prediction")

            if prediction_button:
                prediction = make_prediction(image)
                st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
