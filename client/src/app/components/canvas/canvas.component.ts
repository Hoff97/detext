import { AfterViewInit, Component, ElementRef, EventEmitter, Input, Output, ViewChild } from '@angular/core';
import { fromEvent } from 'rxjs';
import { pairwise, switchMap, takeUntil } from 'rxjs/operators';


type Point = {
  x: number,
  y: number
};

@Component({
  selector: 'app-canvas',
  templateUrl: './canvas.component.html',
  styleUrls: ['./canvas.component.css']
})
export class CanvasComponent implements AfterViewInit {
  // a reference to the canvas element from our template
  @ViewChild('canvas', {static: false}) public canvas: ElementRef;

  // setting a width and height for the canvas
  @Input() public width = 400;
  @Input() public height = 400;
  @Input() public strokeSize = 3;

  @Output() imageChange = new EventEmitter<ImageData>();

  private cx: CanvasRenderingContext2D;

  public ngAfterViewInit() {
    // get the context
    const canvasEl: HTMLCanvasElement = this.canvas.nativeElement;
    this.cx = canvasEl.getContext('2d');

    // set the width and height
    canvasEl.width = this.width;
    canvasEl.height = this.height;

    // we'll implement this method to start capturing mouse events
    this.captureEvents(canvasEl);

    this.clear();
  }

  public clear() {
    this.cx.fillStyle = '#FFF';
    this.cx.beginPath();
    this.cx.rect(0, 0, 500, 500);
    this.cx.fill();
  }

  private getImageContent(): ImageData {
    return this.cx.getImageData(0, 0, this.width, this.height);
  }

  private captureEvents(canvasEl: HTMLCanvasElement) {
    fromEvent(canvasEl, 'mousedown')
      .pipe(
        switchMap((e) => {
          return fromEvent(canvasEl, 'mousemove')
            .pipe(
              takeUntil(fromEvent(canvasEl, 'mouseup')),
              takeUntil(fromEvent(canvasEl, 'mouseleave')),
              pairwise()
            );
        })
      )
      .subscribe((res: [MouseEvent, MouseEvent]) => {
        const rect = canvasEl.getBoundingClientRect();

        const prevPos = {
          x: res[0].clientX - rect.left,
          y: res[0].clientY - rect.top
        };

        const currentPos = {
          x: res[1].clientX - rect.left,
          y: res[1].clientY - rect.top
        };

        this.drawOnCanvas(prevPos, currentPos);
      });

    fromEvent(canvasEl, 'mouseup').subscribe((ev: MouseEvent) => {
      this.imageChange.emit(this.getImageContent());
    });
  }

  private drawOnCanvas(prevPos: Point, currentPos: Point) {
    // incase the context is not set
    if (!this.cx) { return; }

    // set some default properties about the line
    this.cx.lineWidth = this.strokeSize;
    this.cx.lineCap = 'round';
    this.cx.strokeStyle = '#000';

    this.cx.beginPath();

    if (prevPos) {
      this.cx.moveTo(prevPos.x, prevPos.y);
      this.cx.lineTo(currentPos.x, currentPos.y);
      this.cx.stroke();
    }
  }
}
