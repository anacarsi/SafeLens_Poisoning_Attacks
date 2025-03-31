import { Tensor } from 'onnxruntime-web';

const DIMS = [1, 3, 32, 32]; // Required dimensions for final_first_model_cifar10
// const DIMS = [1, 3, 64, 64] // Required dimensions for final_second_model_imagenet

export async function getImageTensorFromPath(path: string, dims: number[] =  DIMS): Promise<Tensor> {
  // 1. load the image  
  var image = await loadImageFromPath(path, dims[2], dims[3]);
  // 2. convert to tensor
  var imageTensor = imageDataToTensor(image, dims);
  // 3. return the tensor
  return imageTensor;
}

async function loadImageFromPath(path: string, width: number = DIMS[2], height: number= DIMS[3]): Promise<ImageData> {
  // Create an image element and load the image
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = path;
  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = reject;
  });

  // Create a canvas and get its context
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('Failed to get 2D context from canvas');
  }

  // Draw and resize the image on the canvas
  ctx.drawImage(img, 0, 0, width, height);

  // Get the image data
  return ctx.getImageData(0, 0, width, height);
}

function imageDataToTensor(image: ImageData, dims: number[]): Tensor {
  // 1. Get buffer data from image and create R, G, and B arrays.
  var imageBufferData = image.data;
  const [redArray, greenArray, blueArray] = new Array(new Array<number>(), new Array<number>(), new Array<number>());

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
    // skip data[i + 3] to filter out the alpha channel
  }

  // 3. Concatenate RGB to transpose [224, 224, 3] -> [3, 224, 224] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  let i, l = transposedData.length; // length, we need this for the loop
  // create the Float32Array size 3 * 224 * 224 for these dimensions output
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (i = 0; i < l; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new Tensor("float32", float32Data, dims);
  return inputTensor;
}

