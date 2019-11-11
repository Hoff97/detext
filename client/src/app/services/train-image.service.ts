import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { environment } from 'src/environments/environment';
import { TrainImage } from '../data/types';

@Injectable({
  providedIn: 'root'
})
export class TrainImageService {

  private urlPrefix = environment.urlPrefix;

  constructor(private http: HttpClient) { }

  create(trainImg: TrainImage) {
    return this.http.post(this.urlPrefix + 'api/image/?format=json', trainImg);
  }
}
