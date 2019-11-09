export interface HasId {
  id?: number;
}

export interface ClassSymbol extends HasId {
  timestamp: string;
  name: string;
  description: string;
  latex: string;
  image: string;
}

export interface Model extends HasId {
  timestamp: string;
  model: string;
}
