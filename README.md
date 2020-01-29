[![Build Status](https://travis-ci.com/Hoff97/detext.svg?branch=develop)](https://travis-ci.com/Hoff97/detext) [![codecov](https://codecov.io/gh/Hoff97/detext/branch/develop/graph/badge.svg)](https://codecov.io/gh/Hoff97/detext)

# Whats this?

This is a simple app for detecting and classifying math symbols. It displays the class of the input of the user
and further information (like the latex code) for it.

# TODO
- [x] Simple classification model for math symbols
- [x] Incorporate model into client using onnx.js
- [x] Add backend for:
  - [x] Adding more training data (by selecting correct classification/new classification)
  - [x] Getting current classification model + class information
  - [x] Periodically/On trigger run relearning
- [x] Make offline usage possible:
  - [x] Download model
  - [x] Download all classes + information
- [ ] Flutter App?
- [x] CI/CD
