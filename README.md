# Computer Vision with TensorFlow

A collection of hands-on Jupyter notebooks for learning computer vision with TensorFlow and Keras. The material is organized into three weeks and covers image classification, object detection, and image segmentation through labs and programming assignments.

## Course map

Open a notebook in GitHub to read it, or use the **Open in Colab** link to run it in Google Colab.

### Week 1 — Transfer learning and object localization

| Notebook | What it covers | Run |
| --- | --- | --- |
| [`w1_lab1.ipynb`](week1/w1_lab1.ipynb) | Transfer learning with cats and dogs | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week1/w1_lab1.ipynb) |
| [`w1_lab2.ipynb`](week1/w1_lab2.ipynb) | CIFAR-10 transfer learning with ResNet50 | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week1/w1_lab2.ipynb) |
| [`w1_lab3.ipynb`](week1/w1_lab3.ipynb) | Image classification and object localization | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week1/w1_lab3.ipynb) |
| [`w1_assignment.ipynb`](week1/w1_assignment.ipynb) | Predicting bounding boxes | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week1/w1_assignment.ipynb) |

### Week 2 — Object detection

| Notebook | What it covers | Run |
| --- | --- | --- |
| [`w2_lab1.ipynb`](week2/w2_lab1.ipynb) | Object detection with TensorFlow Hub | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week2/w2_lab1.ipynb) |
| [`w2_lab2.ipynb`](week2/w2_lab2.ipynb) | Visualizing detected objects and bounding boxes | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week2/w2_lab2.ipynb) |
| [`w2_lab3.ipynb`](week2/w2_lab3.ipynb) | Few-shot object detection with RetinaNet | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week2/w2_lab3.ipynb) |
| [`w2_assignment.ipynb`](week2/w2_assignment.ipynb) | Zombie detection with the TensorFlow Object Detection API | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week2/w2_assignment.ipynb) |

### Week 3 — Image segmentation

| Notebook | What it covers | Run |
| --- | --- | --- |
| [`w3_lab1.ipynb`](week3/w3_lab1.ipynb) | Fully convolutional networks for segmentation | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week3/w3_lab1.ipynb) |
| [`w3_lab2.ipynb`](week3/w3_lab2.ipynb) | U-Net for image segmentation | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week3/w3_lab2.ipynb) |
| [`w3_lab3.ipynb`](week3/w3_lab3.ipynb) | Mask R-CNN instance segmentation demo | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week3/w3_lab3.ipynb) |
| [`w3_assignment.ipynb`](week3/w3_assignment.ipynb) | Handwritten digit segmentation | [Open in Colab](https://colab.research.google.com/github/Alyssonmach/tensorflow-computer-vision/blob/main/week3/w3_assignment.ipynb) |

## Getting started

1. Choose a notebook from the course map above.
2. Open it in Colab, then use **File → Save a copy in Drive** if you want to edit and keep your own copy.
3. Follow the setup instructions in the notebook. Select a GPU or TPU runtime only when the notebook calls for one.

The notebooks have different dependencies and data sources. Many use Colab-specific commands, download datasets or models at runtime, and may need setup changes to run locally or with newer TensorFlow releases. Refer to each notebook's setup cells for its requirements.

## Repository layout

```text
week1/  Transfer learning and object localization
week2/  Object detection
week3/  Image segmentation
```
