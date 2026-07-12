import streamlit as st
import cv2
import numpy as np
import os
import time
import config
from face_detector import FaceDetector
from face_embedder import FaceEmbedder
from face_recognizer import FaceRecognizer
import database 
import user_panel 
import admin_panel 
import home

# Initialize Database
database.init_db()



# Page Config (Dark Theme & Layout)

st.set_page_config(page_title="Find Missing Person", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_models():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    return detector, embedder

detector, embedder = load_models()

# SIDEBAR NAVIGATION

st.sidebar.title("Find Missing Person")
st.sidebar.divider()

panel = st.sidebar.selectbox("Select Panel", ["Home", "User Panel", "Admin Panel"])
st.sidebar.write("Menu")

if panel == "Home":
    menu = None
elif panel == "User Panel":
    menu = st.sidebar.radio("Choose action", ["Register Missing Person"])
elif panel == "Admin Panel":
    menu = st.sidebar.radio("Choose action", [
        "Search by Image", 
        "Live Webcam", 
        "CCTV / Video Detection", 
        "multi Camera Detection", 
        "View All Complaints"
    ])

st.sidebar.divider()
st.sidebar.caption("Version : 1.0")

# ROUTING LOGIC (Connecting to other files)

if panel == "Home":
    home.show_home_page()
# User Module Call
elif panel == "User Panel" and menu == "Register Missing Person":
    user_panel.show_registration_page(detector, embedder)

# Admin Module Call
elif panel == "Admin Panel":
    admin_panel.show_admin_panel(menu, detector, embedder)