import type { NextPage } from 'next';
import Head from 'next/head';
import { useState } from 'react';
import styles from '../styles/Home.module.css';
import ImagePreview from "../components/ImagePreview";
import { FileUpload } from '../components/FileUpload';
import { inferenceONNX } from '../utils/predict';
import { StyledButton } from '../components/Buttons';
import { getImage } from '../utils/randomImage';

const Home: NextPage = () => { 
  const [imagePath, setImagePath] = useState("");
  const [topResultLabel, setLabel] = useState("");
  const [topResultConfidence, setConfidence] = useState("");
  const [inferenceTime, setInferenceTime] = useState("");
  const [stickyContent, setStickyContent] = useState("");

  const changePath = (path: string) => {
    setImagePath(path);
    console.log(path);
    setLabel(`Inferencing...`);
    setConfidence("");
    setInferenceTime("");

    submitInference(path);
  }

  const submitInference = async (path: string) => {
    console.log("submitting");
    var [inferenceResult, inferenceTime] = await inferenceONNX(path);
    console.log("inferenceResult", inferenceResult);
    console.log("inferenceTime", inferenceTime);
    var topResult = inferenceResult[0];

    setLabel(topResult.name.toUpperCase());
    setConfidence(topResult.probability);
    setInferenceTime(`Inference speed: ${inferenceTime} seconds`);
  };

  const showStickyNews = (type: string) => {
    if (type === "cisa") {
      setStickyContent("Stay updated with the latest cybersecurity news from CISA.gov.");
    } else if (type === "research") {
      setStickyContent("Explore cutting-edge research on poisoning attacks in AI systems.");
    }
    const stickyNews = document.getElementById("stickyNews");
    if (stickyNews) {
      stickyNews.style.display = "block";
    }
  };

  const hideStickyNews = () => {
    setStickyContent("");
    const stickyNews = document.getElementById("stickyNews");
    if (stickyNews) {
      stickyNews.style.display = "none";
    }
  };
  
  return (
    <div className={styles.container}>
      <Head>
        <meta name="description" content="SafeLens AI - A cybersecurity tool to identify vulnerabilities in deployed ONNX models. Secure your AI today." />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className={styles.dashboard}>
        <nav className={styles.sidebar}>
          <div className={styles.logoContainer}>
            <img src="/images/logo-safelens.png" alt="SafeLens Logo" className={styles.logo} />
          </div>
          <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/analytics">Analytics</a></li>
            <li><a href="/settings">Settings</a></li>
          </ul>
          <div className={styles.contact}>
            <p>Contact us:</p>
            <p>Email: ana.carsi@gmail.com</p>
            <p>Phone: +1-234-567-890</p>
          </div>
        </nav>

        <div className={styles.mainContent}>
          <div className={styles.upperPane}>
            <div
              className={styles.upperPaneSection}
              data-tooltip="Go to CISA.gov"
              onMouseEnter={() => showStickyNews('cisa')}
              onMouseLeave={hideStickyNews}
            >
              <a href="https://www.cisa.gov/" target="_blank" rel="noopener noreferrer">
                Cybersecurity News
              </a>
            </div>
            <div
              className={styles.upperPaneSection}
              data-tooltip="Explore research articles"
              onMouseEnter={() => showStickyNews('research')}
              onMouseLeave={hideStickyNews}
            >
              <a
                href="https://arxiv.org/search/?query=poisoning+attacks&searchtype=all&abstracts=show&order=-announced_date_first&size=50"
                target="_blank"
                rel="noopener noreferrer"
              >
                Research on Adversarial Attacks
              </a>
            </div>
            <div id="stickyNews" className={styles.stickyNews}>
              {stickyContent}
            </div>
          </div>

          <div className={styles.content}>
            <main className={styles.main}>
              <h1 className={styles.title}>
                SafeLens AI - Cybersecurity for AI Models
              </h1>
              <p className={styles.description}>
                Identify vulnerabilities in deployed ONNX models. Upload an image or explore random samples to evaluate the robustness of your AI system.
              </p>
              <div className={styles.hero}>
                <h1 className={styles.heroTitle}>Welcome to SafeLens</h1>
                <p className={styles.heroSubtitle}>
                  Empowering research with secure AI insights. Upload images and get instant predictions.
                </p>
                <StyledButton
                  onClick={() => {
                    const targetSection = document.getElementById('buttonSection');
                    if (targetSection) {
                      targetSection.scrollIntoView({ behavior: 'smooth' });
                    }
                  }}
                  text="Get Started"
                />
              </div>
              <div id="buttonSection" className={styles.buttonContainer}>
                <FileUpload changePath={changePath} />
                <StyledButton onClick={() => changePath(getImage().value)} text="Random Image" />
              </div>
              <div className={styles.contentContainer}>
                <ImagePreview 
                  width={240} 
                  height={240} 
                  path={imagePath} 
                  topResultLabel={topResultLabel} 
                  topResultConfidence={topResultConfidence} 
                  inferenceTime={inferenceTime} 
                />
              </div>
            </main>
            <footer className={styles.footer}>
              <p>&copy; 2025 SafeLens AI. All rights reserved.</p>
              <ul>
                <li><a href="/about">About Us</a></li>
                <li><a href="/contact">Contact Us</a></li>
              </ul>
            </footer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
