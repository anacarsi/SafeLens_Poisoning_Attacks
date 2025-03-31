import { IMAGE_URLS } from '../data/sample-image-urls';

export const getImage = () => {
  var sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
  var random = Math.floor(Math.random() * (9 - 0 + 1) + 0);
  return sampleImageUrls[random];
}