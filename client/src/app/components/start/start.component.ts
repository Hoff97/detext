import { Component, OnInit } from '@angular/core';
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
}
