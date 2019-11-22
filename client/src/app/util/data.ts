export function base64ToBinary(data: string): Uint8Array {
  const raw = atob(data);
  const rawLength = raw.length;
  const array = new Uint8Array(new ArrayBuffer(rawLength));
  for (let i = 0; i < rawLength; i++) {
    array[i] = raw.charCodeAt(i);
  }
  return array;
}

export function binaryToBase64(arr): string {
  const binstr = Array.prototype.map.call(arr, (ch) => {
      return String.fromCharCode(ch);
  }).join('');
  return btoa(binstr);
}

export function strToBase64(str): string {
  return btoa(str);
}
