import { base64ToBinary, binaryToBase64 } from './data';

describe('dataUtil', () => {
  it('should be able to convert from base 64', () => {
    const testStrs = ['a', 'sdgg94htg'];
    for (const testStr of testStrs) {
      const converted = binaryToBase64(testStr);
      const back = base64ToBinary(converted);
      const backTwice = binaryToBase64(back);

      expect(backTwice).toEqual(converted);
    }
  });
});
