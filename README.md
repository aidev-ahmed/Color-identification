# Color identification

## 📌 Project Overview

This project detects and analyzes the dominant colors in both images and videos using Computer Vision and Machine Learning techniques.

The system extracts the most common colors, calculates their percentages, converts them into RGB and HEX formats, and assigns a readable color name.

For videos, the program samples frames at regular intervals, analyzes each frame, and generates an overall color summary for the entire video.

---

## 🚀 Features

### Image Analysis
- Detect dominant colors in an image
- Extract RGB values
- Generate HEX color codes
- Calculate color percentages
- Assign color names automatically
- Save color palette visualization

### Video Analysis
- Process video files
- Sample frames automatically
- Analyze colors for each sampled frame
- Generate an overall color summary
- Save analyzed frame results
- Create an overall color palette

---

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-Learn (K-Means Clustering)
- Matplotlib

---

## 📂 Project Structure

```text
project/
│
├── main.py
├── test.jpg
├── test.mp4
├── requirements.txt
├── README.md
└── output/
```

---

## ⚙️ Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

---

## ▶️ Usage

### Analyze an Image

```bash
python main.py image.jpg
```

### Analyze a Video

```bash
python main.py video.mp4
```

---

## 📊 Output

The program provides:

- Dominant color names
- RGB values
- HEX values
- Percentage of each color
- Color palette visualization
- Saved output images

Example:

```text
1  Blue        #3A6BFF   RGB(58,107,255)   35.4%
2  White       #F5F5F5   RGB(245,245,245)  21.8%
3  Gray        #8A8A8A   RGB(138,138,138)  15.3%
```

---

## 🧠 Methodology

1. Read image or video file.
2. Convert image from BGR to RGB.
3. Resize image for faster processing.
4. Apply K-Means Clustering.
5. Extract dominant color clusters.
6. Calculate color percentages.
7. Match colors to predefined color names.
8. Display and save the results.

---

## 📸 Example Results

The output includes:

- Original image/frame
- Dominant color palette
- RGB values
- HEX color codes
- Color percentages
- Overall video color summary

---

## 👨‍💻 Author

Ahmed Amr

AI Engineer | Machine Learning & Computer Vision Enthusiast