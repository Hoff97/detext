import { Component, OnInit } from '@angular/core';
import ndarray from 'ndarray';
import { Tensor } from 'onnxjs';
import { InferenceService } from 'src/app/services/inference.service';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.css']
})
export class StartComponent implements OnInit {

  constructor(private inferenceService: InferenceService) { }

  ngOnInit() {}

  async predictClass(image: ImageData) {
    await this.inferenceService.infer(image);
  }

  private preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);

    // Normalize 0-255 to (-1)-1
    ndarray.ops.divseq(dataFromImage, 128.0);
    ndarray.ops.subseq(dataFromImage, 1.0);

    // mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]


    const tensor = new Tensor(new Float32Array(3 * width * height), 'float32', [1, 3, width, height]);
    (tensor.data as Float32Array).set(dataFromImage.data);
    return tensor;
  }
}
