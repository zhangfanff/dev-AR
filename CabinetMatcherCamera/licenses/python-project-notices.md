# Third-party components

- RapidOCR / rapidocr-onnxruntime 1.4.4: RapidAI, Apache License 2.0. https://github.com/RapidAI/RapidOCR
- PP-OCR models: PaddlePaddle/PaddleOCR, Apache License 2.0. https://github.com/PaddlePaddle/PaddleOCR
- ONNX Runtime: Microsoft, MIT License. https://github.com/microsoft/onnxruntime

The ONNX files in models/ are copied unchanged from the published RapidOCR 1.4.4 wheel.
Model hashes and tensor shapes are recorded in models/manifest.json.
The accompanying character dictionary is extracted from recognition-model metadata.

This project also uses NumPy, OpenCV, Pillow, PyYAML, pyclipper and Shapely.
Their licenses remain available in the installed packages and package distributions.
The licenses/ directory retains license texts from the RapidOCR and PaddleOCR source repositories.
