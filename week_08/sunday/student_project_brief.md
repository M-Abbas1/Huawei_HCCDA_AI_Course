# Week 8 CNN Project: Choose a Problem, Build a Classifier

## Project goal

Choose an image-recognition problem that matters to you, find a suitable labeled dataset, and build a small CNN with TensorFlow/Keras that makes predictions on new images. Serve the trained model with a FastAPI endpoint and make a simple Streamlit interface for users to try it.

You may work on the Rock–Paper–Scissors example during class. For your own project, choose a different topic so you practice making the project decisions yourself. The goal is a clear, reproducible end-to-end project—not the largest model or the highest score at any cost.

## Example project areas

These are starting points. You may propose another image classification task if you can explain why it is feasible.

| Area | Example prediction task | Possible dataset to investigate |
|---|---|---|
| Nature | Identify plant species or leaf conditions | PlantVillage |
| Environment | Classify recyclable or household waste | TrashNet or another labeled waste image dataset |
| Transport | Recognize traffic sign categories | German Traffic Sign Recognition Benchmark (GTSRB) |
| Food | Classify food types or identify fruit categories | Food-101 subset or a labeled fruit image dataset |
| Animals | Classify a small set of animal species | Oxford-IIIT Pet or another labeled animal dataset |
| Hand gestures | Recognize a small set of hand signs or gestures | Rock–Paper–Scissors or a suitable sign-language image dataset |
| Everyday objects | Classify common objects | CIFAR-10 or a suitable subset of a larger dataset |

Check dataset access, license, labels, class balance, image quality, and size before committing. Some datasets are large; a manageable subset is acceptable if you document how you selected it. Do not use private or sensitive images without permission. Avoid medical diagnosis tasks for this introductory project.

## Dataset selection checklist

Before selecting a dataset, confirm:

- It contains images and labels for a classification task.
- The classes are clear enough to learn and explain.
- There are enough examples per class for a train/validation/test split.
- No class is missing or extremely underrepresented; if imbalance exists, explain how you will handle it.
- Images can be resized and preprocessed within the available time and hardware.
- You can access and use the data under its license and terms.
- Your test split can be kept separate from training and model selection.
- The dataset includes enough variation to make evaluation meaningful (for example, backgrounds, lighting, angle, or object appearance).

## Required project components

### 1. Problem and data

- State the user-facing problem and who might use the classifier.
- Define the input and the classes the model predicts.
- Describe the dataset source, license, number of images, class counts, image dimensions, and any known limitations.
- Show examples of images and labels.
- Explain your preprocessing and your train/validation/test split.

### 2. Models and training

- Build a simple baseline, such as a small ANN on flattened images or a majority-class predictor.
- Build a CNN with TensorFlow/Keras using convolution and activation layers, plus a classification head. Explain the main layer choices.
- Train using the training split and use the validation split to guide model choices.
- Record settings that affect reproducibility: image size, normalization, batch size, epochs, optimizer, and learning rate.
- Save the trained model and class-name mapping.

### 3. Evaluation and reflection

- Evaluate the final model on the held-out test set.
- Report accuracy and at least one additional useful measure, such as per-class precision/recall or a confusion matrix.
- Compare the CNN to the baseline using the same data split.
- Inspect at least five incorrect predictions or uncertain examples. Discuss what may have caused the errors.
- Include training/validation loss or accuracy curves, if available.
- State limitations and one next improvement you would try.

### 4. FastAPI service

- Load the saved TensorFlow/Keras model once when the service starts.
- Provide a prediction endpoint that accepts an image upload.
- Apply the same preprocessing used during training.
- Return the predicted class and class scores as JSON.
- Handle invalid files or unsupported images with a clear error response.

### 5. Streamlit interface

- Let the user upload an image.
- Display the submitted image and the predicted class.
- Show class scores in a readable way.
- Call the FastAPI service rather than loading a separate copy of the model in the UI.
- Explain what kinds of images the model supports and any important limitations.

## Project proposal template

Complete this before implementation and get instructor feedback.

### A. Project idea

- **Project title:**
- **Area/domain:**
- **Problem in one or two sentences:**
- **Who might use the result, and how?:**
- **Why is image classification appropriate?:**

### B. Prediction task

- **What image will the user provide?:**
- **What classes will the model predict?:**
- **What should the app show after prediction?:**
- **What is outside the scope of this project?:**

### C. Dataset plan

- **Dataset name and source/link:**
- **License or usage terms:**
- **Number of images and number of classes:**
- **Images per class:**
- **Image dimensions and color format (if known):**
- **How will you split the data into train, validation, and test sets?:**
- **What variation or possible bias does the dataset contain?:**
- **What data or access risks could block the project?:**

### D. Model and experiment plan

- **Baseline model:**
- **Initial CNN design:**
- **How will you preprocess images?:**
- **What metric(s) will you report?:**
- **What one experiment will you run (for example, filter count or augmentation)?:**
- **What hardware or time constraints do you expect?:**

### E. App plan

- **What will the FastAPI prediction endpoint accept and return?:**
- **What will the Streamlit user see and be able to do?:**
- **What invalid input should the app handle?:**

### F. Success criteria and risks

- **What result would count as a working project?:**
- **What are the two biggest risks?:**
- **What is your fallback if the dataset or model does not work as expected?:**

## Suggested folder structure

```text
cnn-project/
├── README.md
├── requirements.txt
├── data/                  # data or download instructions; do not commit restricted data
├── models/                # saved model and class mapping
├── notebooks/             # exploration and experiments
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── api/
│   └── main.py             # FastAPI service
└── app/
    └── streamlit_app.py    # Streamlit interface
```

Use TensorFlow/Keras for the model, FastAPI for the prediction service, and Streamlit for the interface. Keep training, prediction preprocessing, API code, and UI code understandable and reproducible.

## Final submission

Submit:

1. The project proposal.
2. Source code and setup/run instructions.
3. A short report covering the problem, dataset, model, evaluation, errors, limitations, and conclusions.
4. The FastAPI endpoint and Streamlit interface, with screenshots or a short demo.
5. A note identifying the dataset source and license, plus any data that cannot be redistributed.

## Assessment rubric

| Category | Weight | What is assessed |
|---|---:|---|
| Problem and dataset choice | 15% | Clear task, suitable dataset, licensing, and honest data description |
| CNN implementation | 25% | Correct preprocessing, architecture, training, and saved model |
| Evaluation and analysis | 25% | Fair split, suitable metrics, baseline comparison, and error analysis |
| FastAPI and Streamlit app | 25% | Working end-to-end prediction with clear input/output and error handling |
| Reproducibility and communication | 10% | Readable code, setup instructions, report, and limitations |

## Important evaluation rule

Do not use the test set to repeatedly tune the model. Make model decisions using training and validation data, then use the test set for the final evaluation. Keep preprocessing consistent between training, evaluation, and API predictions.
