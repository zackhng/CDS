import streamlit as st
from PIL import Image


img_path1 = "img/audio/emotion_count.png"
img1 = Image.open(img_path1)
st.image(img1, caption="Emotion Count")

img_path4 = "img/audio/gender_distribution.png"
img4 = Image.open(img_path4)
st.image(img4, caption="Emotion Count")

img_path2 = "img/audio/emotion_distribution_gender.png"
img2 = Image.open(img_path2)
st.image(img2, caption="Emotion Count")

img_path5 = "img/audio/intensity_distribution.png"
img5 = Image.open(img_path5)
st.image(img5, caption="Emotion Count")

img_path3 = "img/audio/emotion_distribution_intensity.png"
img3 = Image.open(img_path3)
st.image(img3, caption="Emotion Count")

st.markdown(
    '''
## **RMS in dB (Logarithmic Loudness)**

* **dB** = logarithmic scale of amplitude:

  * 0 dB → peak amplitude in the file
  * Negative dB → quieter frames (e.g., -12 dB : Threshold of hearing)
* **RMS** measures average energy per frame

  * Linear RMS → raw amplitude
  * RMS in dB → perceptual loudness

**Reading the graph:**

* **Y-axis:** RMS energy (dB)
* **X-axis:** Frame number (~20–50 ms each)
* **Peaks** → loud syllables / intense speech
* **Valleys** → quiet vowels, consonants, or pauses
* **Benchmark line** (e.g., -12 dB) → identifies “relatively loud” frames

**Emotion patterns:**

* Angry / Fearful → higher peaks, louder
* Calm / Neutral / Sad → lower peaks, quieter
* Happy / Disgust / Surprised → intermediate energy

---

'''
)
img_path10 = "img/audio/rms_emotion.png"
img10 = Image.open(img_path10)
st.image(img10, caption="Emotion Count")

st.markdown("""
            ---
### Rank by Loudness

From highest to lowest RMS:

- **Angry** → 0.0117 (highest)  
- **Fearful** → 0.0087  
- **Happy** → 0.0052  
- **Disgust** → 0.0042  
- **Calm** → 0.0032  
- **Sad** → 0.0030  
- **Neutral** → 0.0021  
- **Surprised** → 0.0018 (lowest)  

**Interpretation:**

- **Angry** and **Fearful** are the loudest emotions, which makes sense — people speak more forcefully or tensely when angry/fearful.  
- **Happy** and **Disgust** are moderate in loudness — happy often slightly louder, disgust can vary.  
- **Neutral, Calm, Sad** → low RMS — these are quieter, less energetic speech.  
- **Surprised** has very low RMS here, which might be due to the sample chosen (some surprised expressions are high-pitched but soft in volume).

---

""")
img_path6 = "img/audio/log_rms_emotion.png"
img6 = Image.open(img_path6)
st.image(img6, caption="Emotion Count")

st.markdown("""
--- 
### RMS Energy Analysis by Emotion

This boxplot shows the **distribution of per-sample RMS energy (`rms_mean`) for each emotion**. RMS energy reflects the **average loudness of each audio file**, and can serve as a feature for emotion recognition.

---

#### 1. Angry

* **Highest median RMS** among all emotions
* **Large spread across samples** (wide box + long whiskers)
* Several high outliers

**Interpretation:**
Samples labeled **angry** tend to be **louder and more variable** than other emotions in the dataset. RMS energy captures the strong vocal intensity typical of anger, though variability indicates some samples are quieter.

---

#### 2. Fearful and Happy

* Median RMS **higher than neutral or calm**, but lower than angry
* Moderate spread with occasional high outliers

**Interpretation:**

* **Fearful** speech can be tense, sometimes louder
* **Happy** speech is energetic and expressive
* Both show **dataset-level variation**, meaning some samples are louder than others

---

#### 3. Neutral and Calm

* **Lowest median RMS**
* Very tight spread among samples
* Few extreme values

**Interpretation:**
Samples labeled **neutral** or **calm** are consistently soft and controlled. RMS energy is effective at distinguishing these low-arousal emotions from high-arousal emotions like angry or happy.

---

#### 4. Surprised, Sad, Disgust

* Median RMS is **moderate**
* Spread is smaller than angry, but larger than calm/neutral

**Interpretation:**
These emotions are less consistently loud. For example:

* **Surprise** may include short bursts of louder speech
* **Sad** speech is mostly quiet but occasionally louder in some samples

---

#### 5. Variability and Feature Implications

* High-variance emotions: **angry, fearful, happy**
* Low-variance emotions: **calm, neutral**

**Interpretation:**

* RMS variation aligns with **arousal level**: high-arousal emotions tend to be louder and more variable across samples.
* RMS can help distinguish **high vs low arousal**, but is insufficient to separate overlapping emotions.

---

#### 6. Outliers

* Represent individual samples with unusually high RMS
* Could reflect speaker differences, recording conditions, or strong vocal emphasis

**Implication:**

* Outliers are normal; extreme values should be considered when **normalizing features** for ML.

---

#### TLDR

1. RMS energy per sample is a **one useful feature** for emotion recognition.
2. High-arousal emotions (angry, happy, fearful) have higher and more variable RMS values in the dataset.
3. Low-arousal emotions (calm, neutral) are softer and more consistent.
4. RMS alone **cannot fully distinguish all emotions** — combining with other features like **pitch, MFCCs, or spectral features** is recommended.

---

""")
img_path11 = "img/audio/rms_energy_dist.png"
img11 = Image.open(img_path11)
st.image(img11, caption="Emotion Count")

st.markdown(
    '''
    ## **Pitch Analysis (Fundamental Frequency, F0)**

**Mean pitch (Hz) per emotion:**

| Emotion   | Mean F0 (Hz) |
| --------- | ------------ |
| Fearful   | 279.5        |
| Angry     | 94.7         |
| Surprised | 127.0        |
| Neutral   | 94.5         |
| Calm      | 91.2         |
| Happy     | 268.8        |
| Sad       | 100.3        |
| Disgust   | 92.7         |

**Interpretation:**

* **High pitch:**

  * Fearful (279.5 Hz) and Happy (268.8 Hz) → consistent with excited, tense, or emotionally heightened speech.
* **Moderate pitch:**

  * Surprised (127 Hz) → short bursts of higher pitch, may vary across samples.
* **Low pitch:**

  * Angry (94.7 Hz), Neutral (94.5 Hz), Calm (91.2 Hz), Sad (100.3 Hz), Disgust (92.7 Hz) → more stable, controlled, or subdued speech.

**Insights:**

* Pitch can be a strong **indicator of emotional state**, especially for high-intensity emotions (fear, happiness).
* Low-intensity or negative emotions (calm, neutral, sad, disgust) tend to cluster in **lower pitch ranges**.
* Surprised has a moderate pitch here — may show **short high-pitched spikes** not captured by the mean alone.

---

'''
)
img_path12 = "img/audio/emotion_pitch.png"
img12 = Image.open(img_path12)
st.image(img12, caption="Emotion Count")

st.markdown(
    '''
    ### **Pitch (F0) Analysis by Emotion**

This boxplot shows the **distribution of mean pitch (F0) per audio file** for each emotion. Each point is **one sample’s average pitch**, reflecting vocal tone. Pitch is informative for **emotion detection**, especially for separating emotions by **excitement**, because:

* High-pitched voices → high excitement (surprised, angry, happy)
* Low-pitched voices → low excitement (calm, neutral)

---

#### **1. High-Pitch / High-Excitement Emotions: Angry, Happy, Surprise**

* **Median pitch:** Surprise slightly higher than Angry ≈ Happy
* **Spread across samples:** Small (based on IQR)
* **Outliers:** Some unusually low-pitch samples (e.g., quieter expressions)

**Interpretation:**

* Samples in these emotions generally have **high average pitch**, making it easier for a model to distinguish them from low-excitement emotions.
* **Surprise** often includes **sudden vocal bursts (screeches)**, increasing mean pitch, explaining its high median.
* Pitch variability and outliers indicate **speaker differences**, so ML models should consider normalization or robust scaling.

---

#### **2. Medium-Pitch / Medium-Excitement Emotions: Disgust, Fearful, Sad**

* **Median pitch:** Slightly above low-pitch emotions but below high-pitch ones
* **Spread across samples:** Moderate (based on IQR)
* **Outliers:** Few low mean-pitch samples and no outlier for fearful

**Interpretation for ML:**

* These emotions are **moderate in vocal frequency**, so pitch alone is insufficient to fully separate them.
* ML models benefit from combining **mean pitch with RMS or spectral features** to resolve overlaps.
* Example: Fearful and Sad may have similar median pitch but differ in RMS variability, helping a classifier distinguish them.

---

#### **3. Low-Pitch / Low-Excitement Emotions: Neutral, Calm**

* **Median pitch:** Lowest among all emotions
* **Spread across samples:** Moderate (based on IQR)
* **Outliers:** Some unusually low mean-pitch samples especially calm

**Interpretation for ML:**

* Low-arousal emotions have **stable, low average pitch**, making them distinguishable from high-arousal emotions.
* Outliers should be handled carefully in preprocessing to prevent skewing features.
* Combined with RMS (low energy), these emotions are **easier for ML models to classify**.

---

#### **6. TLDR**

1. **Mean pitch per audio file is a strong feature for emotion detection**.
2. **High-pitch emotions (Angry, Happy, Surprise):** easier to separate from low-arousal emotions.
3. **Medium-pitch emotions (Disgust, Fearful, Sad):** overlaps exist; need complementary features.
4. **Low-pitch emotions (Neutral, Calm):** consistently low and stable; easier to classify.
5. **Pitch alone is insufficient** — combining with **RMS, MFCCs, spectral features, and temporal features** improves classification accuracy.
6. Outliers reflect natural variation and should be handled in ML pipelines via **normalization or clipping** to reduce feature noise.

---
'''
)
img_path9 = "img/audio/pitch_emotion.png"
img9 = Image.open(img_path9)
st.image(img9, caption="Emotion Count")

st.markdown(
    '''
## Analysis

- Distribution of energy across partial

This computes a Mel spectrogram, which represents:

- X-axis: time
- Y-axis: Mel-scaled frequencies
- Color: energy at that frequency

It shows where the sound energy exists in the frequency spectrum over time.

What it captures?
- Pitch patterns
- Harmonics
- Energy distribution
- Speech intensity

What you see visually?
- Bright areas → strong frequencies
- Dark areas → weak frequencies

This is very useful for visualizing speech structure.

---
'''
)
img_path7 = "img/audio/mel_spectogram.png"
img7 = Image.open(img_path7)
st.image(img7, caption="Emotion Count")

st.markdown(
    '''
MFCCs are not frequencies. They are compressed features derived from the Mel spectrogram.

Pipeline conceptually:

Audio
  
   ↓

Mel Spectrogram
   
   ↓

Log energy
   
   ↓

DCT transform
   
   ↓

MFCC

- MFCC is basically a compact representation of the Mel spectrogram.

What MFCC captures?

- Speech timbre
- Vocal tract characteristics
- Spectral envelope

These are extremely useful for speech recognition and emotion detection.
'''
)
img_path8 = "img/audio/mfcc.png"
img8 = Image.open(img_path8)
st.image(img8, caption="Emotion Count")