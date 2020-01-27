import { Component, OnInit } from '@angular/core';
import { ClassSymbol } from 'src/app/data/types';
import { InferenceService } from 'src/app/services/inference.service';
import { SymbolService } from 'src/app/services/symbol.service';
import { TrainImageService } from 'src/app/services/train-image.service';
import { binaryToBase64 } from 'src/app/util/data';

@Component({
  selector: 'app-start',
  templateUrl: './start.component.html',
  styleUrls: ['./start.component.css']
})
export class StartComponent implements OnInit {

  public loading = false;
  public modelLoading = true;

  public predictions = [];
  public classes = [];

  private img?: ImageData;

  constructor(private inferenceService: InferenceService,
              private trainImageService: TrainImageService,
              private symbolService: SymbolService) {
    this.modelLoading = !this.inferenceService.model;
    this.inferenceService.modelAvailable.subscribe(x => {
      this.modelLoading = false;
    });
  }

  ngOnInit() {
    this.symbolService.getSymbols().subscribe(symbols => {
      this.classes = symbols;
    });
  }

  async predictClass(image: ImageData) {
    this.loading = true;
    this.img = image;
    this.predictions = Array.prototype.slice.call(await this.inferenceService.infer(image));
    this.loading = false;
  }

  correctSelected(cls: ClassSymbol) {
    const b64 = binaryToBase64(this.img.data);

    const trainImg = {
      symbol: cls.id,
      image: b64,
      width: this.img.width,
      height: this.img.height
    };
    this.trainImageService.create(trainImg).subscribe(resp => {
      console.log(resp);
    });
  }

  cleared() {
    this.predictions = [];
  }

  async reloadClasses() {
    this.classes = await this.symbolService.getSymbols().toPromise();
  }
}
