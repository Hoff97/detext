import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

interface Model {
  timestamp: string;
  model: string;
}

@Injectable({
  providedIn: 'root'
})
export class ModelService {

  constructor(private http: HttpClient) { }

  getRecent(): Observable<Model> {
    return this.http.get<Model>('api/model/latest/?format=json');
  }
}
