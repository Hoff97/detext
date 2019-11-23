export interface HasId {
  id?: number;
}

export interface ClassSymbol extends HasId {
  timestamp?: string;
  name: string;
  description: string;
  latex: string;
  image: string;
  imgDatUri?: string;
}

export interface Model extends HasId {
  timestamp: string;
  model: string;
}

export interface TrainImage extends HasId {
  symbol: number;
  image: string;
  timestamp?: string;
  user?: number;
  width: number;
  height: number;
}
