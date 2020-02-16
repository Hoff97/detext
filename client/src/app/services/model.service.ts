import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { from, Observable, of } from 'rxjs';
import { catchError } from 'rxjs/internal/operators/catchError';
import { map } from 'rxjs/internal/operators/map';
import { flatMap } from 'rxjs/operators';
import { environment } from '../../environments/environment';
import { Model } from '../data/types';
import { DbService } from './db.service';
import { Settings, SettingsService } from './settings.service';

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  private key = 'classification-model';

  private urlPrefix = environment.urlPrefix;

  private downloaded = false;

  constructor(private http: HttpClient,
              private dbService: DbService,
              private settingsService: SettingsService) {
    this.settingsService.dataChange.subscribe((data: Settings) => {
      if (data.download && !this.downloaded) {
        this.getRecent().subscribe(x => {
          this.downloaded = data.download;
        });
      }
    });
  }

  getRecent(): Observable<Model|{}> {
    return from(this.dbService.getModel()).pipe(flatMap(model => {
        let url = this.urlPrefix + 'api/model/latest/?format=json';
        if (model) {
          url += `&timestamp=${model.timestamp}`;
        }
        return this.http.get<Model>(url);
      }),
      map((model: Model) => {
        this.dbService.saveModel(model);
        return model;
      }),
      catchError((err) => {
        return of({});
    }));
  }

  getRecentLocal(): Promise<Model> {
    return this.dbService.getModel();
  }

  retrain(): Observable<void> {
    return this.http.post<void>(this.urlPrefix + 'api/model/train/?format=json', {});
  }
}
