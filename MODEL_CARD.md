# Model card: `drawing_classifier.h5`

## Intended use

Educational sketch classification inside the DrawBot ROS course prototype. The model accepts normalized grayscale `28×28×1` inputs and emits scores over 345 labels.

## Preserved facts

- File size: 2,033,664 bytes
- SHA-256: `5ea1c92aacb2d22f505797bdb51acf4f21867cf35fccca35770314e7000a7927`
- Embedded topology: three convolution/max-pooling blocks, a 512-unit dense layer and a 345-unit softmax output
- Label list: `src/juno_sketch_tutor/config/classes.txt`

## Provenance boundary

The model file was included in the original course deliverable. Its exact source checkpoint, training command, random seed, train/validation split and held-out evaluation are not preserved in this repository. `train_drawing_classifier.py` documents a compatible architecture and optional training path; it does not prove that it reproduces these exact weights.

The label space follows Google's [Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset), which contains 345 categories and is released under CC BY 4.0.

## Evaluation status

No defensible accuracy, calibration, fairness or per-class robustness result is available from the repository artifacts. Unit tests inspect topology and validate surrounding software with simulated probability vectors; they do not measure model quality.

## Limitations

- Camera sketches can differ substantially from Quick, Draw! bitmaps.
- Confidence scores are uncalibrated and should not be interpreted as probabilities of correctness.
- Lighting, paper texture, framing and line thickness can change preprocessing output.
- The model is suitable for classroom demonstration, not safety-critical autonomy.
