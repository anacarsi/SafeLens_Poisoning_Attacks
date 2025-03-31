import Image from 'next/image'
import styles from '../styles/ImagePreview.module.css'

interface Props {
  height: number;
  width: number;
  path: string;
  topResultLabel: string;
  topResultConfidence: string;
  inferenceTime: string;
}

const ImagePreview = (props: Props) => {
  var draw = (
    <>
    </>
  )
  if (props.path) {
    draw = (
      <div className={styles.previewContainer}>
        <Image
          src={props.path}
          width={props.width}
          height={props.height}
          alt="Picture to be inferenced"
        />
        <div className={styles.labelsContainer}>
          <span>{props.topResultLabel} {props.topResultConfidence}</span>
          <span>{props.inferenceTime}</span>
        </div>
      </div>
    )
  }
  return (
    <>
      {draw}
    </>
  )
};

export default ImagePreview;
