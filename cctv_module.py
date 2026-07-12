import streamlit as st
import cv2
import numpy as np
import database
from face_recognizer import FaceRecognizer
import email_notifier


def process_video_footage(detector, embedder):
    """
    Logic to process Network IP Cameras and CCTV RTSP streams

    """
    st.title("Live CCTV Integration")
    st.write(
        "Connect directly to your IP Camera or DVR/NVR using an RTSP or HTTP stream URL."
    )
    st.divider()

    # 1. Input CCTV URL
    cctv_url = st.text_input(
        "Enter CCTV Stream URL (RTSP/HTTP)",
        placeholder="e.g., rtsp://admin:12345@192.168.1.50:554/stream",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        # Start and stop using checkbox
        run_cctv = st.checkbox("Start CCTV Stream")

    if run_cctv:
        if not cctv_url:
            st.error("Please enter a valid CCTV URL to connect.")
            return

        saved_embeddings = database.get_all_embeddings()

        if not saved_embeddings:
            st.warning("Database is empty! Please register a missing person first.")
            return

        recognizer = FaceRecognizer(saved_embeddings)
        frame_window = st.image([])

        # 2. Pass URL to OpenCV
        cap = cv2.VideoCapture(cctv_url)

        emailed_persons = set()
        frame_counter = 0
        frame_skip_rate = 30  # To reduce load on Laptop/Server
        current_face_data = []

        try:
            with st.spinner("Connecting to CCTV Network..."):
                while run_cctv:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Connection Lost! Please check the camera URL, network, or password.")
                        break

                    frame_counter += 1
                    email_frame = frame.copy()

                    # 3. AI Detection & Tracking Logic
                    if frame_counter % frame_skip_rate == 0:
                        detected_faces = detector.detect_faces(frame)
                        new_face_data = []

                        for face in detected_faces:
                            x1, y1, x2, y2 = face["box"]
                            new_center_x, new_center_y = (x1 + x2) // 2, (y1 + y2) // 2

                            matched_name, matched_score = None, None

                            # Centroid Tracking logic to check if same person
                            for old_face in current_face_data:
                                ox1, oy1, ox2, oy2 = old_face["box"]
                                old_center_x, old_center_y = (ox1 + ox2) // 2, (oy1 + oy2) // 2
                                dist_sq = (new_center_x - old_center_x) ** 2 + ( new_center_y - old_center_y) ** 2

                                if dist_sq < 10000:
                                    matched_name = old_face["name"]
                                    matched_score = old_face["score"]
                                    break

                            if matched_name is not None:
                                name, score = matched_name, matched_score
                            else:
                                # New face found, run InsightFace
                                new_emb = embedder.get_embedding(face["crop"])
                                if new_emb is not None:
                                    name, score = recognizer.identify_person(new_emb)
                                else:
                                    name, score = "Unknown", 0.0

                                # Email Alert Generator
                                if name != "Unknown" and name not in emailed_persons:
                                    receiver_email = database.get_email_by_name(name)
                                    if receiver_email:
                                        st.toast(
                                            f"Alert! {name} spotted on CCTV stream.",icon="🚨",)

                                        # Hiding sensitive network details in location name, just showing IP if present
                                        safe_location = (
                                            "CCTV IP: " + cctv_url.split("@")[-1]
                                            if "@" in cctv_url
                                            else "Network Camera"
                                        )

                                        color = (0, 255, 0)
                                        cv2.rectangle(
                                            email_frame, (x1, y1), (x2, y2), color, 2)
                                        cv2.putText(
                                            email_frame,
                                            f"{name} ({score:.2f})",
                                            (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.7,
                                            color,
                                            2,
                                        )

                                        email_notifier.trigger_email_alert(
                                            name,
                                            receiver_email,
                                            email_frame,
                                            location=safe_location,
                                        )
                                        emailed_persons.add(name)

                            new_face_data.append(
                                {"box": (x1, y1, x2, y2), "name": name, "score": score}
                            )

                        current_face_data = new_face_data

                    # 4. Box Drawing Loop
                    for face_info in current_face_data:
                        x1, y1, x2, y2 = face_info["box"]
                        name = face_info["name"]
                        score = face_info["score"]

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = (
                            f"{name} ({score:.2f})" if name != "Unknown" else "Unknown"
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

                    # Update UI
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(frame_rgb, width="stretch")

        finally:
            cap.release()
