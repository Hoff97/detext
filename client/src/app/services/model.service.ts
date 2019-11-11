import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { catchError } from 'rxjs/internal/operators/catchError';
import { map } from 'rxjs/internal/operators/map';
import { environment } from '../../environments/environment';
import { Model } from '../data/types';
import { DbService } from './db.service';

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  private key = 'classification-model';

  private urlPrefix = environment.urlPrefix;

  constructor(private http: HttpClient, private dbService: DbService) { }

  getRecent(): Observable<Model> {
    return this.http.get<Model>(this.urlPrefix + 'api/model/latest/?format=json').pipe(
      map((model: Model) => {
        this.dbService.saveModel(model);
        return model;
      }),
      catchError((err) => {
        return this.dbService.getModel();
      })
    );
  }
}
