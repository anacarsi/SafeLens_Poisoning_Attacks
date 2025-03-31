"use client";
import styles from '../styles/FileUpload.module.css';

interface Props {
  /** Function to handle the validated image URL */
  changePath: Function;
}

// Constants for image validation
const MAX_DIMENSIONS = 32; // Required dimensions for final_first_model_cifar10
// const MAX_DIMENSIONS = 64; // Required dimensions for final_second_model_imagenet
const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];

export const FileUpload = (props: Props) => {
  /**
   * Validates the uploaded image file
   * @param file - The file to validate
   * @returns Promise<boolean> - Resolves to true if image meets requirements
   */
  const validateImage = (file: File): Promise<boolean> => {
    return new Promise((resolve) => {
      // Validate file type
      if (!ALLOWED_FILE_TYPES.includes(file.type)) {
        alert('Please upload a JPG or PNG image.');
        resolve(false);
        return;
      }

      const img = new Image();
      
      // Handle successful image load
      img.onload = () => {
        // Validate exact dimensions required for neural network
        if (img.width !== MAX_DIMENSIONS || img.height !== MAX_DIMENSIONS) {
          alert(`Image must be exactly ${MAX_DIMENSIONS}x${MAX_DIMENSIONS} pixels.`);
          URL.revokeObjectURL(img.src);
          resolve(false);
          return;
        }

        // Initialize canvas for image processing
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          alert('Failed to process image.');
          resolve(false);
          return;
        }

        // Draw image to canvas for potential pixel data access
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // Get image data for RGB validation
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        
        // Cleanup and resolve
        URL.revokeObjectURL(img.src);
        resolve(true);
      };

      // Handle image load failure
      img.onerror = () => {
        alert('Failed to load image. Please try another file.');
        URL.revokeObjectURL(img.src);
        resolve(false);
      };

      // Begin image loading process
      img.src = URL.createObjectURL(file);
    });
  };

  return (
    <div className={styles.uploadContainer}>
      <label
        htmlFor="file"
        className={styles.uploadButton}
      >
        Choose File
      </label>
      <input
        id="file"
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        className={styles.hiddenInput}
        onChange={async (e) => {
          const files = e.target.files;
          if (files && files[0]) {
            // Validate image before creating blob URL
            if (await validateImage(files[0])) {
              const blobUrl = URL.createObjectURL(files[0]);
              props.changePath(blobUrl);
            }
          }
        }}
      />
    </div>
  );
};