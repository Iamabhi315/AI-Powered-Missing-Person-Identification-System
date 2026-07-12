import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import database
from face_recognizer import FaceRecognizer
import email_notifier
import cctv_module
import multi_camera_module

def show_admin_panel(menu, detector, embedder):
    
    # 1. LIVE WEBCAM MODULE

    if menu == "Live Webcam":
        st.title("Live Webcam Detection")
        st.divider()
        st.info("Press the button below to start detection.")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            run_camera = st.checkbox("Start Webcam", key="live_cam")
            
        if run_camera:
            saved_embeddings = database.get_all_embeddings()
            
            if not saved_embeddings:
                st.warning("Database is empty. Please register a missing person first.")
            else:
                recognizer = FaceRecognizer(saved_embeddings)
                frame_window = st.image([])
                
                cap = cv2.VideoCapture(0)
                emailed_persons = set()
                
                frame_counter = 0
                frame_skip_rate = 30  
                current_face_data = [] # Faces from the previous frame will be stored here
                
                try:
                    while run_camera:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Cannot access camera.")
                            break
                        
                        frame_counter += 1
                        email_frame = frame.copy()
                        
                        if frame_counter % frame_skip_rate == 0:
                            detected_faces = detector.detect_faces(frame)
                            new_face_data = [] # Newly updated faces
                            
                            for face in detected_faces:
                                x1, y1, x2, y2 = face['box']
                                
                                # 1. Calculate the center (centroid) of the new face
                                new_center_x = (x1 + x2) // 2
                                new_center_y = (y1 + y2) // 2
                                
                                matched_name = None
                                matched_score = None
                                
                                # 2. Check whether this face is already present in the camera view
                                for old_face in current_face_data:
                                    ox1, oy1, ox2, oy2 = old_face['box']
                                    old_center_x = (ox1 + ox2) // 2
                                    old_center_y = (oy1 + oy2) // 2
                                    
                                    # Calculate the distance using the Pythagorean theorem
                                    dist_sq = (new_center_x - old_center_x)**2 + (new_center_y - old_center_y)**2
                                    
                                    # If the distance is less than 10000 (about 100 pixels), it is the same person
                                    if dist_sq < 10000:
                                        matched_name = old_face['name']
                                        matched_score = old_face['score']
                                        break
                                
                                # 3. If it is the same person, do not run heavy AI again
                                if matched_name is not None:
                                    name = matched_name
                                    score = matched_score
                                
                                # 4. Embedd only when new person comes in camera
                                else:
                                    face_crop = face['crop']
                                    new_emb = embedder.get_embedding(face_crop)
                                    
                                    if new_emb is not None:
                                        name, score = recognizer.identify_person(new_emb)
                                    else:
                                        name, score = "Unknown", 0.0
                                        
                                    # Email will be checked for newly detected person only
                                    if name != "Unknown" and name not in emailed_persons:
                                        receiver_email = database.get_email_by_name(name)
                                        if receiver_email:
                                            st.toast(f"Match Found: {name}! Sending email alert...", icon="🚨")
                                            import email_notifier
                                            camera_location = "Live Webcam (Admin Panel)"
                                            
                                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                            cv2.rectangle(email_frame, (x1, y1), (x2, y2), color, 2)
                                            label = f"{name} ({score:.2f})"
                                            cv2.putText(email_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                            
                                            try:
                                                email_notifier.trigger_email_alert(name, receiver_email, email_frame, location=camera_location)
                                                emailed_persons.add(name)
                                            except Exception as exc:
                                                st.error(f"Email not sent for {name}: {exc}")
                                        else:
                                            st.warning(f"No receiver email found for {name}.")
                                
                                # Update current data to compare next frame
                                new_face_data.append({
                                    'box': (x1, y1, x2, y2),
                                    'name': name,
                                    'score': score
                                })
                            
                            # Replace old data to new data
                            current_face_data = new_face_data

                        # Drawing Loop (iterate on every frame , to keep movement smooth)
                        for face_info in current_face_data:
                            x1, y1, x2, y2 = face_info['box']
                            name = face_info['name']
                            score = face_info['score']
                            
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame_rgb)
                        
                finally:
                    cap.release()


    # 2. SEARCH BY IMAGE MODULE
    elif menu == "Search by Image":
        st.title("Search by Image")
        st.write("Upload a CCTV screenshot or photo to search against the missing persons database.")
        st.divider()
        
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            saved_embeddings = database.get_all_embeddings()
            if not saved_embeddings:
                st.warning("Database is empty! Register someone first.")
            else:
                recognizer = FaceRecognizer(saved_embeddings)
                with st.spinner("Analyzing image..."):
                    detected_faces = detector.detect_faces(image)
                    
                    if len(detected_faces) == 0:
                        st.error("No faces detected in the uploaded image.")
                    else:
                        st.success(f"Detected {len(detected_faces)} face(s). Matching with database...")
                        for face in detected_faces:
                            x1, y1, x2, y2 = face['box']
                            face_crop = face['crop']
                            new_emb = embedder.get_embedding(face_crop)
                            
                            if new_emb is not None:
                                name, score = recognizer.identify_person(new_emb)
                                
                                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                                label = f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
                                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        st.image(image_rgb, caption="Search Result", width="stretch")

    # 3. CCTV / VIDEO DETECTION MODULE
    elif menu == "CCTV / Video Detection":
        cctv_module.process_video_footage(detector, embedder)


    # 4. MULTI CAMERA DETECTION MODULE
    elif menu == "multi Camera Detection":
        multi_camera_module.show_multi_camera_panel(detector, embedder)

    # 5. VIEW ALL COMPLAINTS MODULE
    elif menu == "View All Complaints":
        st.title("Registered Missing Persons")
        st.divider()
        
        # Connect to the database and display the table
        conn = sqlite3.connect(database.DB_NAME)
        # Remove the embedding column from the query because it is binary data and cannot be displayed easily
        df = pd.read_sql_query("SELECT id, full_name, gender, age, email, address FROM missing_persons", conn)
        
        if df.empty:
            st.info("No complaints registered yet.")
        else:
            # hide_index=True prevents row numbers from appearing, keeping the UI clean
            st.dataframe(df, width="stretch", hide_index=True)

            st.divider()
            st.subheader("Delete a Person Record")
            person_names = database.get_all_person_names()

            if person_names:
                selected_person = st.selectbox("Select a person to delete", person_names)
                if st.button("Delete Selected Person", type="primary"):
                    if database.delete_person(selected_person):
                        st.success(f"Deleted record for {selected_person}.")
                        st.rerun()
                    else:
                        st.error("Could not delete the selected person.")
            else:
                st.info("No person records available to delete.")
        
        conn.close()
        
    # 5. OTHER MODULES (Under Development)
    else:
        st.title(menu)
        st.info(f"The '{menu}' module is currently under development.")