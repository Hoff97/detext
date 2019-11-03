import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/internal/operators/catchError';
import { map } from 'rxjs/internal/operators/map';
import { DbService } from './db.service';

export interface Model {
  timestamp: string;
  model: string;
  id?: number;
}

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  private key = 'classification-model';

  constructor(private http: HttpClient, private dbService: DbService) { }

  getRecent(): Observable<Model> {
    return this.http.get<Model>('api/model/latest/?format=json').pipe(
      map((model: Model) => {
        this.dbService.saveModel(model);
        return model;
      }),
      catchError((err) => {
        return this.dbService.getModel();
      })
    );
  }

  private loadModel(): Model {
    return JSON.parse(localStorage.getItem(this.key));
  }
}
