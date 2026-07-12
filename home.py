import streamlit as st


def show_home_page():
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
        padding: 2rem 2rem; border-radius: 18px; margin-bottom: 1.5rem; color: white;">
            <h1 style="margin-bottom: 0.3rem;">Find Missing Person</h1>
            <p style="font-size: 1.1rem; margin-top: 0; opacity: 0.95;">
                AI-powered face recognition for faster and smarter identification.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="background: #111827; padding: 1.4rem; border-radius: 16px; border: 1px solid #374151; margin-bottom: 1rem;">
        <h3 style="margin-top: 0; color: #60a5fa;">Welcome Section</h3>
        <p style="color: #e5e7eb; margin-bottom: 0.8rem;">
            This platform helps you register missing persons, search through uploaded images, and monitor live camera feeds for possible matches using advanced AI.
        </p>
        <div style="display: flex; flex-wrap: wrap; gap: 0.7rem;">
            <span style="background: #1d4ed8; color: white; padding: 0.4rem 0.7rem; border-radius: 999px; font-size: 0.9rem;">Face Registration</span>
            <span style="background: #0f766e; color: white; padding: 0.4rem 0.7rem; border-radius: 999px; font-size: 0.9rem;">Image Search</span>
            <span style="background: #7c3aed; color: white; padding: 0.4rem 0.7rem; border-radius: 999px; font-size: 0.9rem;">Live CCTV Monitoring</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div style="background: #0f172a; padding: 1.2rem; border-radius: 14px; border: 1px solid #334155; margin-bottom: 1rem;">
            <h3 style="margin-top: 0; color: #34d399;">Key features</h3>
            <ul style="color: #cbd5e1; line-height: 1.7;">
                <li>Register missing persons with face images</li>
                <li>Search by uploaded image</li>
                <li>Detect matches using webcam or CCTV footage</li>
                <li>Send email alerts when a strong match is found</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #1f2937; padding: 1.2rem; border-radius: 14px; border: 1px solid #4b5563;">
            <h3 style="margin-top: 0; color: #fbbf24;">Get started</h3>
            <p style="color: #e5e7eb;">Choose a panel from the sidebar to begin your task.</p>
            <div style="background: #111827; padding: 0.8rem; border-radius: 10px; color: #93c5fd;">
                User Panel → Register a missing person<br>
                Admin Panel → Search and monitor cases
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.info("Select a panel from the sidebar to continue.")
