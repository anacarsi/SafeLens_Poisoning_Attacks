import styles from '../styles/FileUpload.module.css';

interface Props {
  onClick: () => void;
  text: string;
}

export const StyledButton = (props: Props) => {
  return (
    <div className={styles.uploadContainer}>
      <button
        onClick={props.onClick}
        className={styles.uploadButton}
      >
        {props.text}
      </button>
    </div>
  );
};
