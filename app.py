import streamlit as st
import cv2
from detector import RoadSignDetector
from time import sleep



def main():
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='font-size: 28px;'>AQROAD: AI-powered real-time road sign detection and recognition.</h1>", unsafe_allow_html=True)
    
    detector = RoadSignDetector()

    col1, col2 = st.columns([0.4, 0.6])  
    
    with col1:

        video_placeholder = st.empty()
    
    with col2:

        sign_name = st.empty()
        description = st.empty()
        confidence = st.empty()
        
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera")
            break
            
        detections = detector.detect_signs(frame)

        frame = detector.draw_detections(frame, detections)

        if detections:

            best_detection = max(detections, key=lambda x: x['confidence'])
            
            sign_name.markdown(f"### Detected Sign:\n{best_detection['name']}")
            description.markdown(f"### Description:\n{best_detection['description']}")
            confidence.markdown(f"### Confidence:\n{best_detection['confidence']:.2f}")
          
        else:
            sign_name.markdown("### No signs detected")
            description.markdown("### Description:\nNo sign detected")
            confidence.markdown("### Confidence: N/A")
        
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        sleep(0.1)
    
    cap.release()

if __name__ == "__main__":
    main()