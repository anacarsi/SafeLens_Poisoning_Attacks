// Language: typescript
// Path: react-next\utils\predict.ts
import { getImageTensorFromPath } from './imageHelper';
import { runOnnxModel } from './modelHelper';

export async function inferenceONNX(path: string): Promise<[any,number]> {
  console.log(path)
  // 1. Convert image to tensor
  const imageTensor = await getImageTensorFromPath(path);
  // 2. Run model
  const [predictions, inferenceTime] = await runOnnxModel(imageTensor);
  return [predictions, inferenceTime];
}
