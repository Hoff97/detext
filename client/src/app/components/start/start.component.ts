import { Component, OnInit } from '@angular/core';
import { InferenceService } from 'src/app/services/inference.service';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.css']
})
export class StartComponent implements OnInit {

  public loading = false;

  public predictions = [];
  public classes = [];

  constructor(private inferenceService: InferenceService) { }

  ngOnInit() {}

  async predictClass(image: ImageData) {
    this.loading = true;
    this.predictions = Array.prototype.slice.call(await this.inferenceService.infer(image));
    this.classes = this.inferenceService.getClasses();
    this.loading = false;
  }
}
