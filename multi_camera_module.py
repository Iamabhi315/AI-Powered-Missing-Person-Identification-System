import streamlit as st
import cv2
import numpy as np
import database
from face_recognizer import FaceRecognizer
import email_notifier

def show_multi_camera_panel(detector, embedder):
    st.title("Multi Camera Detection")
    st.write("Connect and monitor multiple CCTV/RTSP streams simultaneously on a single dashboard.")
    st.divider()

    # 1. URL for two different cameras
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        url_cam1 = st.text_input("Enter Camera 1 URL (RTSP/HTTP/0 for Webcam)", placeholder="e.g., 0")
    with col_input2:
        url_cam2 = st.text_input("Enter Camera 2 URL (RTSP/HTTP)", placeholder="e.g., rtsp://...")

    run_multi = st.checkbox("Start Multi-Camera Monitoring")

    if run_multi:
        if not url_cam1 or not url_cam2:
            st.error("Please enter URLs for both cameras to start multi-stream.")
            return

        saved_embeddings = database.get_all_embeddings()
        if not saved_embeddings:
            st.warning("Database is empty! Register someone first.")
            return

        recognizer = FaceRecognizer(saved_embeddings)

        # 2. Layout grid setup for displaying both cameras side by side
        col_view1, col_view2 = st.columns(2)
        with col_view1:
            st.caption("🎥 Camera 1 Feed")
            frame_window1 = st.image([])
        with col_view2:
            st.caption("🎥 Camera 2 Feed")
            frame_window2 = st.image([])

        # Initialize OpenCV video capture objects
        # If the user enters '0', it must be converted to the integer 0 for webcam access
        input1 = int(url_cam1) if url_cam1.isdigit() else url_cam1
        input2 = int(url_cam2) if url_cam2.isdigit() else url_cam2

        cap1 = cv2.VideoCapture(input1)
        cap2 = cv2.VideoCapture(input2)

        emailed_persons = set()
        
        # Separate tracking variables for each camera
        frame_counter = 0
        frame_skip_rate = 30
        face_data_cam1 = []
        face_data_cam2 = []

        try:
            with st.spinner("Connecting to both camera networks..."):
                while run_multi:
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()

                    if not ret1 and not ret2:
                        st.error("Lost connection to both cameras!")
                        break

                    frame_counter += 1

                    # CAMERA 1 PROCESSING
                    if ret1:
                        email_frame1 = frame1.copy()
                        if frame_counter % frame_skip_rate == 0:
                            detected_faces1 = detector.detect_faces(frame1)
                            face_data_cam1 = []
                            
                            for face in detected_faces1:
                                x1, y1, x2, y2 = face['box']
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                matched_name = None
                                
                                for old_face in face_data_cam1:
                                    ox1, oy1, ox2, oy2 = old_face['box']
                                    ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                                    if ((cx - ocx)**2 + (cy - ocy)**2) < 10000:
                                        matched_name = old_face['name']
                                        score = old_face['score']
                                        break
                                
                                if matched_name is not None:
                                    name = matched_name
                                else:
                                    new_emb = embedder.get_embedding(face['crop'])
                                    name, score = recognizer.identify_person(new_emb) if new_emb is not None else ("Unknown", 0.0)
                                    
                                    if name != "Unknown" and name not in emailed_persons:
                                        receiver_email = database.get_email_by_name(name)
                                        if receiver_email:
                                            st.toast(f"🚨 Match on Camera 1: {name}!", icon="👤")
                                            cv2.rectangle(email_frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            email_notifier.trigger_email_alert(name, receiver_email, email_frame1, location="Multi-Cam: Camera 1")
                                            emailed_persons.add(name)

                                face_data_cam1.append({'box': (x1, y1, x2, y2), 'name': name, 'score': score})

                        # Draw bounding boxes for Camera 1
                        for face_info in face_data_cam1:
                            x1, y1, x2, y2 = face_info['box']
                            name = face_info['name']
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame1, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        frame_window1.image(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), width="stretch")

                    # CAMERA 2 PROCESSING
                    if ret2:
                        email_frame2 = frame2.copy()
                        if frame_counter % frame_skip_rate == 0:
                            detected_faces2 = detector.detect_faces(frame2)
                            face_data_cam2 = []
                            
                            for face in detected_faces2:
                                x1, y1, x2, y2 = face['box']
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                matched_name = None
                                
                                for old_face in face_data_cam2:
                                    ox1, oy1, ox2, oy2 = old_face['box']
                                    ocx, ocy = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                                    if ((cx - ocx)**2 + (cy - ocy)**2) < 10000:
                                        matched_name = old_face['name']
                                        score = old_face['score']
                                        break
                                
                                if matched_name is not None:
                                    name = matched_name
                                else:
                                    new_emb = embedder.get_embedding(face['crop'])
                                    name, score = recognizer.identify_person(new_emb) if new_emb is not None else ("Unknown", 0.0)
                                    
                                    if name != "Unknown" and name not in emailed_persons:
                                        receiver_email = database.get_email_by_name(name)
                                        if receiver_email:
                                            st.toast(f"🚨 Match on Camera 2: {name}!", icon="👤")
                                            cv2.rectangle(email_frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            email_notifier.trigger_email_alert(name, receiver_email, email_frame2, location="Multi-Cam: Camera 2")
                                            emailed_persons.add(name)

                                face_data_cam2.append({'box': (x1, y1, x2, y2), 'name': name, 'score': score})

                        # Draw bounding boxes for Camera 2
                        for face_info in face_data_cam2:
                            x1, y1, x2, y2 = face_info['box']
                            name = face_info['name']
                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame2, f"{name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        frame_window2.image(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), width="stretch")

        finally:
            cap1.release()
            cap2.release()