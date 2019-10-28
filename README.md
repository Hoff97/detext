# Whats this?

This is a simple app for detecting and classifying math symbols. It displays the class of the input of the user
and further information (like the latex code) for it

# TODO
- [x] Simple classification model for math symbols
- [x] Incorporate model into client using onnx.js
- [ ] Add backend for:
  - [ ] Adding more training data (by selecting correct classification/new classification)
  - [ ] Getting current classification model + class information
  - [ ] Periodically/On trigger run relearning
- [ ] Make offline usage possible:
  - [ ] Download model
  - [ ] Download all classes + information
- [ ] Flutter App?
- [ ] CI/CD