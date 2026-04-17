# Human Activity Recognition on Edge AI using Quantized CNN-GRU

A TinyML project that trains, quantizes, and deploys a CNN-GRU model for real-time Human Activity Recognition (HAR) on edge devices — directly mirroring the architecture and goals of livestock behavior monitoring systems like TinyCowNet.

---

## Project Overview

This project classifies 6 human activities from raw accelerometer data using a lightweight CNN-GRU neural network, then compresses it via INT8 quantization for deployment on microcontrollers (ESP32/Arduino class devices). The pipeline covers data preprocessing, model training, quantization, and TFLite edge inference.

**Dataset:** UCI HAR Dataset — accelerometer (x, y, z) time-series data  
**Activities:** Jogging · Walking · Upstairs · Downstairs · Sitting · Standing  
**Hardware Target:** ESP32 / Arduino Nano 33 BLE Sense (TFLite Micro compatible)

---

## Architecture Evolution

### Why not plain LSTM or GRU?

Three architectures were evaluated before arriving at the final model:

#### LSTM (Baseline)
Trained a 2-layer LSTM (64 units each) on 200-timestep windows.

| Metric | Float32 | Quantized |
|--------|---------|-----------|
| Model Size | 217.26 KB | 87.73 KB |
| Accuracy | 98.19% | 40.63% |
| Inference | 23.55 ms | 20.93 ms |
| Compression | 2.48x | |

**Problem:** INT8 quantization collapsed accuracy from 98% to 40%. LSTM ops are not fully supported by `TFLITE_BUILTINS_INT8`, requiring `SELECT_TF_OPS` fallback which breaks weight quantization.

---

#### GRU with unroll=True
Switched to GRU and enabled `unroll=True` to make the model quantization-compatible (static graph required for INT8).

| Metric | Float32 | INT8 |
|--------|---------|------|
| Model Size | 1457.45 KB | 6319.91 KB |
| Accuracy | 99.00% | 97.55% |
| Inference | 9.50 ms | 13.20 ms |
| Compression | 0.23x | |

**Problem:** `unroll=True` replicates GRU weights across every timestep. With 200 timesteps, this caused a **4x size explosion** — 6319 KB is completely unusable on microcontrollers.

---

#### CNN-GRU Hybrid (Final Model ✅)
Used a 1D-CNN front-end to compress the 200-timestep sequence down to ~23 timesteps before the GRU layer. This eliminates the unrolling explosion while keeping the model quantization-friendly.

```
Input (200, 3)
→ Conv1D(32, kernel=5) + MaxPool(4)   → (49, 32)
→ Conv1D(64, kernel=3) + MaxPool(2)   → (23, 64)
→ GRU(32, unroll=True)                → (32,)
→ Dense(6, softmax)                   → 6 classes
```

| Metric | Float32 | INT8 |
|--------|---------|------|
| Model Size | 145.10 KB | **42.42 KB** |
| Accuracy | 98.75% | **98.46%** |
| Avg Inference | 0.436 ms | 1.889 ms |
| Size Reduction | **3.42x** | |

**Result:** 3.42x compression, only 0.29% accuracy drop, 42 KB fits on ESP32.

---

## Quantization

INT8 post-training quantization using TensorFlow Lite with a representative dataset:

```python
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
converter_q.representative_dataset = representative_dataset
converter_q.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter_q._experimental_lower_tensor_list_ops = False
converter_q.inference_input_type = tf.float32
converter_q.inference_output_type = tf.float32
```

Internal weights and ops are quantized to INT8 while keeping float32 I/O for compatibility.

---

## Edge Inference Demo

The quantized `.tflite` model runs on-device using the TFLite interpreter, simulating deployment on an ESP32:

```
=================================
 CNN-GRU HAR - ESP32 Simulation
 Edge AI Inference Demo
=================================
Model : CNN-GRU INT8 Quantized
Size  : 42.42 KB
Accuracy: 98.46%
---------------------------------
Input:      Jogging sample
Predicted:  Jogging
True Label: Jogging
Confidence: 97.3%
Inference:  142.0 us
Result:     CORRECT ✓
---------------------------------
```

---

##  Live Actions Monitoring

This project is a direct analog to edge AI behavior monitoring systems:

| Research Requirement | This Project |
|---|---|
| Optimized RNN for behavior classification | CNN-GRU hybrid with INT8 quantization |
| Memory-efficient edge deployment | 42.42 KB model — ESP32 compatible |
| Sensor data (accelerometer) | UCI HAR accelerometer (x, y, z axes) |
| Activity classification | 6-class: jogging, walking, upstairs, downstairs, sitting, standing |
| Quantization techniques | INT8 PTQ with 3.42x compression, <0.3% accuracy loss |
| Architectural tradeoff analysis | LSTM → GRU → CNN-GRU evolution documented |

---

## Requirements

```
tensorflow >= 2.10
numpy
pandas
scikit-learn
scipy
```

---

## File Structure

```
├── model_training.ipynb       # Full training pipeline
├── cnn_gru_har_model.keras    # Trained Keras model
├── cnn_gru_har_float32.tflite # Float32 TFLite model (145.10 KB)
├── cnn_gru_int8.tflite        # INT8 quantized model  (42.42 KB)
```
