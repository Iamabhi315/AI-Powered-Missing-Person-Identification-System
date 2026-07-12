import streamlit as st
import cv2
import numpy as np
import os
import config
import database

def show_registration_page(detector, embedder):
    """Handles the user panel registration form and its logic."""
    st.title("Register Missing Person")
    st.divider()
    
    # Form layout
    with st.form("registration_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            full_name = st.text_input("Full Name")
            email = st.text_input("Email Address")
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=1, max_value=120, value=20)
            
        address = st.text_area("Address")
        
        st.write("Upload Face Images")
        uploaded_files = st.file_uploader(
            "Upload at least 3 clear face images for better accuracy.", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        submit_btn = st.form_submit_button("Submit Registration")
        
        if submit_btn:
            full_name = full_name.strip() # Remove extra spaces
            
            if not full_name:
                st.error("Full Name is required!")
            elif not uploaded_files:
                st.error("Please upload at least one image.")
            else:
                with st.spinner("Saving images and analyzing face..."):
                    
                    # 1. Create a new folder for the person inside the image directory
                    person_folder = os.path.join(config.IMAGE_ROOT, full_name)
                    os.makedirs(person_folder, exist_ok=True)
                    
                    valid_embedding = None
                    saved_images_count = 0
                    
                    # 2. Save each uploaded image into the person's folder
                    for idx, uploaded_file in enumerate(uploaded_files):
                        image_bytes = uploaded_file.read()
                        file_extension = uploaded_file.name.split(".")[-1]
                        file_path = os.path.join(person_folder, f"img_{idx}.{file_extension}")
                        
                        with open(file_path, "wb") as f:
                            f.write(image_bytes)
                        saved_images_count += 1
                        
                        # 3. Extract a face for the AI model
                        if valid_embedding is None:
                            file_bytes_np = np.asarray(bytearray(image_bytes), dtype=np.uint8)
                            image = cv2.imdecode(file_bytes_np, 1)
                            
                            faces = detector.detect_faces(image)
                            if len(faces) > 0:
                                valid_embedding = embedder.get_embedding(faces[0]['crop'])

                    # 4. Save the data to the database
                    if valid_embedding is not None:
                        database.insert_person(full_name, gender, email, age, address, valid_embedding)
                        st.success(f"Registration successful for {full_name}!")
                        st.info(f"📁 {saved_images_count} images successfully saved in '{person_folder}'")
                    else:
                        st.warning(f"Images saved in '{person_folder}', but no clear face was detected for the AI database. Please upload clearer photos.")