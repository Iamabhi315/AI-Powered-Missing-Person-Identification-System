import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
import threading
from datetime import datetime, timedelta
import config

_email_lock = threading.Lock()
_recent_alerts = {}


def _send_email_task(person_name, receiver_email, frame, location):
    server = None
    try:
        print(f"[INFO] Background task started: Sending email to {receiver_email}...")

        now = datetime.now()
        dt_string = now.strftime("%d-%b-%Y at %I:%M:%S %p")

        msg = MIMEMultipart()
        msg['Subject'] = f"🚨 URGENT: {person_name} has been Spotted!"
        msg['From'] = config.SENDER_EMAIL
        msg['To'] = receiver_email

        body = f"""Hello,

Our AI system has successfully detected {person_name} on our camera feed.

📍 DETECTION DETAILS:
-------------------------------------------------
• Person Name     : {person_name}
• Date & Time     : {dt_string}
• Camera Location : {location}
-------------------------------------------------

Please find the attached live screenshot for reference.

Regards,
AI Face Recognition System"""

        msg.attach(MIMEText(body, 'plain'))

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            image_attachment = MIMEImage(buffer.tobytes(), name=f"{person_name}_spotted.jpg")
            msg.attach(image_attachment)
        else:
            print("[WARNING] Image encode failed, sending without image.")

        if not config.SENDER_EMAIL or not config.SENDER_PASSWORD:
            print("[WARNING] Email not sent because sender credentials are missing. Set SENDER_EMAIL and SENDER_PASSWORD.")
            return

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=15)
        server.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print(f"Email successfully delivered to {receiver_email} for {person_name}")
    except (smtplib.SMTPException, OSError, TimeoutError, ConnectionResetError) as e:
        print(f"[WARNING] Email delivery failed: {e}")
    finally:
        if server is not None:
            try:
                server.close()
            except Exception:
                pass


def trigger_email_alert(person_name, receiver_email, frame, location):
    key = (person_name.lower(), receiver_email.lower())
    now = datetime.now()

    with _email_lock:
        last_sent = _recent_alerts.get(key)
        if last_sent and now - last_sent < timedelta(minutes=1):
            return False
        _recent_alerts[key] = now

    email_thread = threading.Thread(
        target=_send_email_task,
        args=(person_name, receiver_email, frame, location),
        daemon=True
    )
    email_thread.start()
    return True