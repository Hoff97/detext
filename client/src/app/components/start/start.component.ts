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
  public modelUpdating = true;

  public predictions = [];
  public uncertainties = [];
  public classes = [];

  private img?: ImageData;

  constructor(private inferenceService: InferenceService,
              private trainImageService: TrainImageService,
              private symbolService: SymbolService) {
    this.modelLoading = !this.inferenceService.model;
    this.modelUpdating = this.inferenceService.modelUpdating;
    this.inferenceService.modelAvailable.subscribe(x => {
      this.modelLoading = false;
    });
    this.inferenceService.updating.subscribe(x => {
      this.modelUpdating = false;
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
    const [predictions, uncertainty] = await this.inferenceService.infer(image);
    this.predictions = Array.prototype.slice.call(predictions);
    this.uncertainties = Array.prototype.slice.call(uncertainty);
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
    this.uncertainties = [];
  }

  async reloadClasses() {
    this.classes = await this.symbolService.getSymbols().toPromise();
  }
}
