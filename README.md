# Virtual Try-On Web Application

This repository contains a fully automated, production-ready Virtual Try-On web application. It builds upon the foundational research of DCI-VTON but significantly extends and fixes the original implementation to provide a stable, end-to-end user experience.

## Enhancements & Fixes
The original codebase was largely incomplete, lacking a user interface and assuming datasets were strictly pre-processed before use. To make this an interactive app, the following major components were built from scratch or heavily refactored:

1. **End-to-End Automated Pipeline:** Built a custom orchestrator that seamlessly connects the distinct phases over a single run: SegFormer (person parsing & preprocessing), PF-AFN (initial warping), and the final DCI-VTON (diffusion generation).
2. **Interactive Web Interface:** Developed a complete web interface (`web_app/`) offering features like an interactive before/after interactive slider, dynamic image zooming, and real-time processing time tracking.
3. **Optimized Inference & Stability:** The original code frequently suffered from CUDA Out-Of-Memory (OOM) bottlenecks and crashes. These were fixed by correctly isolating GPU memory contexts.
4. **Latency Improvements:** Altered the code architecture to aggressively cache heavy models in-memory between runs and reduced DDIM sampling steps.
5. **One-Click Public Hosting:** Added `ngrok` integration (`start_web_server.bat`) so the development environment can easily be streamed and showcased on the web.

## Under the Hood: The Preprocessing Engine
A major addition to this project is the fully customized, real-time image parsing engine (`image_parsing.py`) which automatically generates the strict dataset format that the DCI-VTON model requires. It runs the following pipeline on every user image:
1. **Human Parsing (SegFormer):** Uses `mattmdjaga/segformer_b2_clothes` to extract a semantic segmentation map of the user, extracting precise body and clothing labels.
2. **Agnostic Parsing:** Strips away the user's current upper clothing, leaving an "agnostic" body silhouette representing only the face, arms, and lower body.
3. **Skeleton Extraction (MediaPipe):** Uses MediaPipe Pose estimation to rapidly generate robust 18-point OpenPose-format keypoints, avoiding the heavy overhead of native OpenPose.
4. **Pose Rendering:** Renders a 2D skeletal map required by the diffusion mechanism.
5. **Cloth Masking:** Generates a clean binary silhouette of the target garment.
6. **DensePose Approximation:** Uses OpenCV GrabCut and regional bounding boxes to synthesize a fast pseudo-IUV depth map of the user.
7. **Upper-Body Garment Warping:** Automatically calculates bounding vectors to scale, position, and static-warp the target cloth to fit perfectly over the user's torso region.

## Demonstrations & Recordings
Live video demonstrations, walkthroughs, and screen recordings of the Virtual Try-On application in action can be located locally at:
`C:\Users\ADMIN\Videos\Screen Recordings`

## Usage
Simply run the web server initialization script `start_web_server.bat` to start tracking and loading models. Access the server either through `localhost` or via the public tunnel URL provided by the script during runtime.
